import torch
import numpy as np
import scipy as scp
# use string to generate \mathcal{L} set of loci letter
import string
# used to compute all recombination events
from itertools import chain, combinations
from torch.distributions import Dirichlet
from scipy.stats import multivariate_normal
from mysrc.utils import SimplexToFromRd

class ModelWrightFisher():
    ## Help file and description of the inputs ##
    def __init__(self, loss_fn, n_parameter, population_size, generation, generation_interval,
                 recomb_param, haplotype_freq, dominance_param = None, rng = None):
        self.loss_fn = loss_fn
        self.epistasis = False
        self.epistasis_alpha = 8
        self.L = n_parameter
        self.population_size = population_size
        self.generation = generation
        self.generation_interval = generation_interval
        if dominance_param is None:
            self.dominance_param = [.5 for ind in range(self.L)]
        else:
            self.dominance_param = dominance_param
        self.haplotype_freq = np.array(haplotype_freq)
        self.haplo = False
        self.recomb_param = recomb_param
        if rng is None:
            self.rng = np.random.RandomState(19)
        else:
            self.rng = rng

        ## Some partitions and their maps pre-calculated
        ## Calculate list of all possible non-empty partitions of {1,2,..,L}
        P_partitions = self.compute_IJ_partitions(r=self.recomb_param)
        self.P_partitions = P_partitions
        ## Now calculate i,j maps to i',j' for all partitions in P_partitions where i and j can be 1,..,2**L
        ij_map_Partition = np.zeros(shape=(2 ** self.L, 2 ** self.L, len(self.P_partitions), 2), dtype=int)
        for ind_iter_i in range(2 ** self.L):
            for ind_iter_j in range(2 ** self.L):
                for ind_P in range(len(self.P_partitions)):
                    ij_map_Partition[ind_iter_i, ind_iter_j, ind_P, 0] = \
                        int(self.get_haplotype_index(ind_iter_i, ind_iter_j, P_partitions[ind_P][0][0], self.L))
                    ij_map_Partition[ind_iter_i, ind_iter_j, ind_P, 1] = \
                        int(self.get_haplotype_index(ind_iter_j, ind_iter_i, P_partitions[ind_P][0][0], self.L))
        self.ij_map_Partition = ij_map_Partition


    def approx_llhd(self, parameters, data, n_ensemble = 10, if_grad = False):
        # Convert data to torch tensor
        #data = torch.Tensor(np.array(data))
        #!!!! This model can't be used with autograd!!!!!

        simulations = self.forward_simulation(parameters, n_data=n_ensemble)
        loss = -1 * self.loss_fn(simulations, data)
        if if_grad:
            raise RuntimeError("Loss is not differentiable for this model.")
        else:
            return loss.item()

    def forward_simulation(self, parameters, n_data=1):

        parameters_numpy = parameters.detach().numpy()
        selection_coeff = parameters_numpy[:self.L]
        # #### Initial Haplotype Frequency ####
        # hapl_freq_z = np.array(parameters_numpy[self.L:self.L+pow(2, self.L)-1])
        # #### Dominance parameter ####
        # h_all_z = parameters_numpy[self.L+pow(2, self.L)-1:2*self.L+pow(2, self.L)-1]
        # #### epistasis alpha parameter ####
        # epistasis_alpha_z = parameters_numpy[2*self.L+pow(2, self.L)-1:2*self.L+pow(2, self.L)]
        # ### Recombination Rate
        # r_all = parameters_numpy[2*self.L+pow(2, self.L):]
        # for every wf sim with fixed L I have the same fitness matrix W(Hij) and
        # same possible recombination events I, summarised into the list of P_partitions
        W_ij = self.compute_fitness_matrix_L_loci(s=selection_coeff, h=self.dominance_param, alpha=self.epistasis_alpha)
        self.W_ij = W_ij
        # however rho_ij(t) has to be updated within the simulator

        if len(self.generation_interval) == 1:
            timestep_generation = np.linspace(0, self.generation,
                                          int(self.generation / self.generation_interval[0]) + 1, dtype=int)
        else:
            timestep_generation = np.array([int(0)]+[int(x) for x in self.generation_interval])

        # Do the actual forward simulation
        vector_of_k_samples = self.wf_sim(fitness_matrix=W_ij,
                                          population_size=self.population_size,
                                          initial_hapl_freq=self.haplotype_freq,
                                          last_gen=self.generation,
                                          timestep_generation=timestep_generation,
                                          num_forward_simulations=n_data, rng=np.random.RandomState(19))
        # Format the output to obey API
        return torch.stack(vector_of_k_samples, dim=1)

    # There are a series of functions that can be done once and used as defaults for
    # the rest of the simulations
    def compute_fitness_matrix_L_loci(self, s, h, alpha):
        """
        Computes the fitness matrix W(H_ij) with a general number of L loci.

        Args:
            s (list/numpy.ndarray (L,) ): list (L,) containing the selection coefficients.
            h (list/numpy.ndarray (L,) ): list (L,) containing the dominanc parameters.

        Returns:
            numpy.ndarray: A (2^L, 2^L) array containing the fitness values for pair of haplotypes.
        """
        if self.L == 1:
            W = np.array([[1, 1 - s[0] * h[0]],
                          [1 - s[0] * h[0], 1 - s[0]]])
        else:
            w_l_tensor = []
            for l in range(self.L):
                # create single (2x2) w_l matrices
                w_l_array = np.array([[1, 1 - s[l] * h[l]],
                                      [1 - s[l] * h[l], 1 - s[l]]])

                # Create a tuple containing the reshaping for matrix multiplication
                L_ones_l_list = [1] * self.L
                L_ones_l_list[l] = 2
                L_ones_l_tuple = tuple(L_ones_l_list + L_ones_l_list)
                w_l_array = w_l_array.reshape(L_ones_l_tuple)

                w_l_tensor.append(w_l_array)

            # matrix multiplication between w_l matrices
            W = 1
            for l in range(self.L):
                W = W * w_l_tensor[l]
            W = W.reshape(2 ** self.L, 2 ** self.L)

        if self.epistasis:
            if self.L == 1:
                Epis_X = np.array([[np.exp(0), np.exp(s * h)],
                                   [np.exp(s * h), np.exp(s)]])
            else:
                epis_X_l_tensor = []
                for l in range(self.L):
                    # create single (2x2) epis_X_l matrices
                    epis_X_l_array = np.array([[np.exp(0), np.exp(s[l] * h[l])],
                                               [np.exp(s[l] * h[l]), np.exp(s[l])]])

                    # Create a tuple containing the reshaping for matrix multiplication
                    L_ones_l_list = [1] * self.L
                    L_ones_l_list[l] = 2
                    L_ones_l_tuple = tuple(L_ones_l_list + L_ones_l_list)
                    epis_X_l_array = epis_X_l_array.reshape(L_ones_l_tuple)

                    epis_X_l_tensor.append(epis_X_l_array)

                # matrix multiplication between w_l matrices
                Epis_X = 1
                for l in range(self.L):
                    Epis_X = Epis_X * epis_X_l_tensor[l]
                Epis_X = alpha * np.power(np.log(Epis_X.reshape(2 ** self.L, 2 ** self.L)) / sum(s), 2)

        if self.epistasis:
            W = W * Epis_X

        return W


    def powerset(self, set):
        """
        Auxiliary function computing the powerset of a given set.

        Args:
            set (list): a non empty list of elements in the set.

        Returns:
            function to compute the powerset.

        e.g.
            powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        """
        s = list(set)  # allows duplicate elements
        return chain.from_iterable(combinations(s, r) for r in np.arange(1, len(s) + 1))


    def compute_IJ_partitions(self, r):
        """
        Computes the all possible pairs of recombination events IJ, partitioning
        the loci set into two exhaustive and non-empty sets for a general L-loci scenario.

        Args:
            r (list): A list of recombination rates for pairwise loci-break.

        Returns:
            A list of all partitions. Each partition is a tuple containing the two elements:
            the list of I, J sets (each I and J is a list with loci letters) and a list of
            one containing the conputed r_I coefficient.

        NOTE: The number of possible partitions, for L>1, can be computed exactly with
        the following formula: sum_{l = 1, L-1} binom(L-1, l)
        Code available:
        \{
            P_len = 0
            for l in range(L-1):
                P_len += scipy.special.binom(L-1, l)
            P_len = int(P_len)
        \}
        """
        L = self.L

        # call the letters I will use for the loci
        loci_alphabet = string.ascii_uppercase[:L]

        # create list of recombination rates with letter of the loci pair
        R_loci_values = []
        for l in range(L - 1):
            r_loci_value = [[loci_alphabet[l], loci_alphabet[l + 1]], [r[l]]]
            R_loci_values.append(r_loci_value)

        L_set = list(loci_alphabet)

        # Creating all possible Recombination events (I)
        L_set_noA = L_set.copy()
        L_set_noA.remove("A")
        all_other_partitions = self.powerset(L_set_noA)
        I_sets = [["A"]] + [["A"] + list(elem) for elem in all_other_partitions]
        I_sets = I_sets[:-1]

        P_len = len(I_sets)
        P_partitions = []
        for pp in range(P_len):
            current_J_set = L_set.copy()
            for e in range(len(I_sets[pp])):
                current_J_set.remove(I_sets[pp][e])

            p_part = [I_sets[pp], current_J_set]

            P_partitions.append(p_part)

        # Compute recomination rates according to events I
        for pp in range(P_len):
            # create an empty vector of length  = len(r) = L-1
            r_counts = np.zeros(L - 1, dtype=int)
            # for each event I, count how many elements of I each r coefficient involves (max 2)
            for e in range(len(I_sets[pp])):
                for r_rate in range(L - 1):
                    if I_sets[pp][e] in R_loci_values[r_rate][0]:
                        r_counts[r_rate] += 1
            # coefficient r enters the multiplication only if it includes 1 of the elements,
            # not if 0 or 2, so just set the 2s to 0.
            r_counts = [0 if rr == 2 else rr for rr in r_counts]
            if np.sum(r_counts) == 1:
                # only containing one r
                r_I = np.sum(np.array(r_counts) * r)
            else:
                # need to compute product of r coefficients
                r_I = np.prod(np.array(r_counts) * r)

            P_partitions[pp] = P_partitions[pp], [r_I]

        return P_partitions


    # end of preliminary functions

    def get_hapl_index_for_loci(self, L):
        """
        Auxiliary function producing a list of L lists,
        each indicating which indices of the hapl freq vector need to be considered
        to compute the freq of allele_1 of each locus.
        """

        length = 2 ** L
        sublists = [[] for _ in range(L)]

        for i in range(L):
            step = 2 ** (L - 1 - i)  # Number of elements to select and skip
            j = 0
            while j < length:
                # Select `step` elements
                sublists[i].extend(range(j, min(j + step, length)))
                # Skip `step` elements
                j += 2 * step
        return sublists


    def calculate_one_allele_freq(self, x_hapl_freq):
        """
        Takes a vector of haplotypes frequencies with L loci (each with two alleles)
        and returns the allele frequencies of type-1 allele (i.e.the wildtype).
        (For L = 1 there is a simpler one but I kept this general)
        """

        # we can deduce L by the lenght of the matrix
        L = int(np.log2(len(x_hapl_freq)))

        hapl_index_for_loci = self.get_hapl_index_for_loci(L)

        allele_frequencies = [np.sum(x_hapl_freq[index_locus_al]) for index_locus_al in hapl_index_for_loci]
        # I noticed a rounding error, with e-10 decimals
        return allele_frequencies


    def calculate_allele_freq_X(self, X_hapl_freq):
        """"
        Computes allele frequencies over multiple timepoints.
        """
        if len(X_hapl_freq.shape) == 1:
            ALL_allele_freq = self.calculate_one_allele_freq(X_hapl_freq)
        else:
            timepoints = X_hapl_freq.shape[1]

            ALL_allele_freq = []
            for tt in range(timepoints):
                # print(self.calculate_one_allele_freq(X_hapl_freq[:,tt]))
                ALL_allele_freq.append(self.calculate_one_allele_freq(X_hapl_freq[:, tt]))
        # print(ALL_allele_freq)
        return np.vstack(ALL_allele_freq)


    def wf_sim(self, fitness_matrix,
               population_size, initial_hapl_freq, last_gen, timestep_generation, num_forward_simulations, rng):
        """
        Simulate n replicates of a discrete wf trajectory.

        Returns n matrices of shape (t, L) for the allele frequencies at each timepoint,
        Here, rng is passing the fixed random state to each of the simulations.
        """
        # define equally spaced discrete timepoints for the generations
        timestep_generation_normalized = timestep_generation / max(timestep_generation)

        # print(timestep_generation)

        result_allele = []
        result_haplo = []

        for iter_simul in range(num_forward_simulations):
            # if num_forward_simulations>1:
            #    current_rng_r = rng_rep[n]
            # else:
            haplo_freq_trajectory = self.compute_haplo_freq_traj(w=fitness_matrix,
                                                                 x_initial=initial_hapl_freq,
                                                                 gen=last_gen,
                                                                 population_size=population_size,
                                                                 rng=rng)[timestep_generation, :]
            # haplo
            result_haplo.append(haplo_freq_trajectory.T)

            # allele
            allele_freq_trajectory = self.calculate_allele_freq_X(haplo_freq_trajectory.T)
            #print(allele_freq_trajectory.shape)
            #print(np.hstack((timestep_generation.reshape(-1,1), allele_freq_trajectory)).shape)
            allele_freq_trajectory = torch.tensor(np.hstack((timestep_generation_normalized.reshape(-1,1), allele_freq_trajectory)))
            #allele_freq_trajectory = torch.tensor(allele_freq_trajectory)
            #print(allele_freq_trajectory.shape)
            #allele_freq_trajectory = allele_freq_trajectory.T.reshape(-1)
            # print(allele_freq_trajectory.shape)
            # allele_freq_trajectory = np.insert(allele_freq_trajectory,
            #                                    0, len(timestep_generation))
            # print(allele_freq_trajectory.shape)
            result_allele.append(allele_freq_trajectory)


        #result_haplo = [np.array([x]).reshape(-1, ) for x in result_haplo]
        #result_haplo = [np.insert(np.around(k, 3), 0, int(last_gen / generation_interval) + 1) for k in result_haplo]

        #if self.haplo:
        #    return result_haplo, result_allele
        #else:
        return result_allele


    def compute_haplo_freq_traj(self, w, x_initial, rng, gen=60, population_size=1000):
        """
        Calculates the haplotype frequencies over time under genetic drift given
        the haplotype frequencies tilde{x}_i after accounting for recombination
        and selection.
        This function is independent of the number of loci, L.

        Args:
            x (numpy.ndarray): An array containing the starting frequencies of the haplotypes.

        Returns:
            numpy.ndarray: A 2D array with dimensions (t, 8) representing the haplotype frequencies at each timepoint.
        """
        freq = np.zeros((gen + 1, 2 ** self.L))
        freq[0, :] = x_initial
        x = x_initial
        for iter_gen in range(gen):
            RHO_t = self.compute_rho_ij(x, self.P_partitions, self.L)
            x_tilde = self.compute_x_tilde_hapl_freq(w=w,x=x,rho=RHO_t)
            # already normalised (but just in case)
            # x_tilde = x_tilde/np.sum(x_tilde)
            x = rng.multinomial(2 * population_size, x_tilde) / (2 * population_size)
            freq[iter_gen + 1, :] = x
        # print(freq.shape)
        return freq


    def index_to_haplotype(self, index, L):
        """
        Given an index, this function converts it into the corresponding
        haplotype for a general number of loci.
        The index ranges from 0 to 2^L, and the haplotype ranges from [0]*L to [1]*L.


        Args:
            index (int): index of the current haplotype.
            L (int): number of loci.
        Returns:
            tuple: A (L,) tuple with 0-1s with each zeros indicating allele for
            each corresponding positional locus.
        """

        if index >= 2 ** L:
            raise RuntimeError(
                'Index cannot be larger than max(2^L-1).')

        # Write it in a general way
        haplotype_list = [index // (2 ** (L - 1))]
        for i in np.arange(1, L - 1):
            current_locus_index = (index % 2 ** (L - i)) // (2 ** (L - i - 1))
            haplotype_list.append(current_locus_index)
        # Last one is fixed
        haplotype_list.append(index % 2)

        return tuple(haplotype_list)


    def get_haplotype_index(self, h1, h2, I, L):
        """
        This function takes two haplotype indices (h1 and h2) and a set I (decomposition of L).
        It returns the index of the haplotype formed by combining components of h1 and h2
        based on the decomposition I.
        It works with L>1 as assessed in higher function.

        Args:
            h1 (int): index of haplotype 1.
            h2 (int): index of haplotype 2.
            I (list): list with the letters of loci contained in the I decomposition.
            L (int): number of loci.
        Returns:
            int: th index of the resulting haplotype
            from the recombination of h1 and hy according to I.

        NOTE: the output using (h1, h2, I, L) is the same as using (h1, h2, J, L),
        with I-J coming from same partition.
        """

        # Convert the indices h1 and h2 to their corresponding haplotypes.
        h1_haplotype = self.index_to_haplotype(h1, L)
        h2_haplotype = self.index_to_haplotype(h2, L)

        # Create the resulting haplotype by choosing components from h1_haplotype if the
        # index is in I, and from h2_haplotype if the index is not in I.
        # hence: H_(i_I, j_J)
        upper_alphabet = list(string.ascii_uppercase[:L])
        # technically we know hoe to obtain H_(j_I, i_J) in an analogous way

        resulting_hapl = [h1_haplotype[ii] if locus_l in I else h2_haplotype[ii] for ii, locus_l in
                          enumerate(upper_alphabet)]

        # Convert the resulting haplotype back to an index, which is returned.
        resulting_index = 0
        for l in range(L):
            resulting_index = resulting_index + resulting_hapl[l] * (2 ** (L - (l + 1)))

        return resulting_index


    def compute_rho_ij(self, x, P_partitions, L):
        """
        Computes the recombination matrix RHO_ij(t).
        Works with L>1 as specified in upper function.

        Args:
            x (numpy.ndarray): An array containing the haplotype frequencies x(t).
            P_partitions (list): list of all possible recombination events
            L (int): number of loci.
        Returns:
            numpy.ndarray: A (2^l, 2^L) array representing the recombination term
            for each haplotype.

        NOTE: the returning RHO_ij is a squared symmetric matrix with 0's in the main diagonal.
        """
        # Initialize an array of (2^L)x(2^L) for all rho's
        rho_ij = np.zeros((2 ** L, 2 ** L))

        if L > 1:
            # If L = 1, all rho_ij's stay zero.

            # Iterate over all couplings i-j
            for i in range(2 ** L):
                for j in np.arange(i + 1, 2 ** L):
                    if len(P_partitions) > 1:  # L>2
                        event_I_to_rho_ij = np.zeros(len(P_partitions))
                        for pp in range(len(P_partitions)):
                            #current_I = P_partitions[pp][0][0]  # select the IJ partition, specifically I
                            current_r_I = P_partitions[pp][1][0]  # select r_I

                            # Get the indices of the haplotypes formed by combining components
                            # # of haplotypes i and j based on the decomposition I.
                            #i_Ij_J_index = self.get_haplotype_index(i, j, current_I, L)
                            #j_Ii_J_index = self.get_haplotype_index(j, i, current_I, L)

                            i_Ij_J_index = self.ij_map_Partition[i,j,pp,0]
                            j_Ii_J_index = self.ij_map_Partition[i,j,pp,1]

                            # contribution of current I
                            event_I_to_rho_ij[pp] = current_r_I * (x[i] * x[j] - x[i_Ij_J_index] * x[j_Ii_J_index])
                        rho_ij[i, j] = np.sum(event_I_to_rho_ij)
                        rho_ij[j, i] = rho_ij[i, j]

                    # When you just have two loci, L = 2
                    else:
                        #current_I = P_partitions[0][0][0]  # select the IJ part, and I
                        current_r_I = P_partitions[0][1][0]  # select r_I

                        # consider both I-J swaps
                        #i_Ij_J_index = self.get_haplotype_index(i, j, current_I, L)
                        #j_Ii_J_index = self.get_haplotype_index(j, i, current_I, L)
                        i_Ij_J_index = self.ij_map_Partition[i, j, 0, 0]
                        j_Ii_J_index = self.ij_map_Partition[i, j, 0, 1]

                        # contribution of current I
                        rho_ij[i, j] = current_r_I * (x[i] * x[j] - x[i_Ij_J_index] * x[j_Ii_J_index])
                        rho_ij[j, i] = current_r_I * (x[i] * x[j] - x[i_Ij_J_index] * x[j_Ii_J_index])
        return rho_ij


    def compute_x_tilde_hapl_freq(self, w, x, rho):
        """
        Computes the next generation haplotyple frequencies tilde{x}_(t+1)
        accounting for recombination and selection.
        """

        # could be more efficient to include it up there as it would be the same i - j loop
        # as r_ij
        x_tilde = np.zeros(x.shape)
        for i in range(len(x_tilde)):
            unnormalised_x_tilde_i = np.zeros(x.shape)
            for j in range(len(x_tilde)):
                unnormalised_x_tilde_i[j] = w[i, j] * (x[i] * x[j] - rho[i, j])
            x_tilde[i] = np.sum(unnormalised_x_tilde_i)
        # Normalize the recombination/selection haplotype frequencies x_tilde_i by
        # the sum of the outer product of the haplotype frequencies multiplied by the fitness matrix.

        common_denom = np.sum(w * np.outer(x, x))
        x_tilde = x_tilde / common_denom
        return x_tilde


    def transform_Rd_to_Simplex(self, We):
        """
        This function applies a reverse transformation to the realisation of a Dirichlet sample in Rd
        back to the original sample space.

        In our context:
        Computes the initial haplotype frequencies (x_0 in the 2^L simplex)
        using 2^L -1 dimensional random variable in Rd, denoted as -We- \in R 2^{L}-1
            Args:
                We:         2^{L}-1 realisations from Dir random variables in real line (-inf, +inf)
            Returns:
                x_cand_0:   The candidate vector of initial haplotype frequencies x_0 in R 2^{L}
        """

        We = np.array(We)
        m = len(We) + 1
        Z_vector = np.zeros(m - 1)
        x_dir = np.zeros(m)
        for k in np.arange(1, m):
            # logit 1/(m-k+1)
            p_mk1 = 1 / (m - k + 1)
            z_arg = We[k - 1] + np.log(p_mk1 / (1 - p_mk1))
            # inv logit
            Z_vector[k - 1] = 1 / (1 + np.exp(-z_arg))

            if k == 1:
                x_dir[k - 1] = Z_vector[k - 1]
            else:
                x_dir[k - 1] = (1 - np.sum(x_dir[:k])) * Z_vector[k - 1]

        # k = m
        x_dir[-1] = 1 - np.sum(x_dir[:-1])

        return x_dir


    def transform_Rd_to_Constrained(self, values_in_real, prior_ranges):
        """
        This function applies an inverse transformation to the realisation of d-dimensional samples
        in Rd back to the uniform range.

        Transform back samples from the Rd into d-uniform with their own ranges.
            Args:
                values_in_real (d-dim array):       Realisation from d uniform distributions in real line range (R^d).
                                                    supports atomic values
                prior_ranges (dx2 array):           Array of prior ranges of the uniforms.
            Returns:
                values_in_unif:                     The output vector of values in d-uniforms
        """
        if isinstance(values_in_real, int):
            values_in_unif = prior_ranges[0] + (prior_ranges[1] - prior_ranges[0]) / (1 + np.exp(-values_in_real))
            return list(values_in_unif)[0]
        elif len(values_in_real) == 1:
            values_in_unif = prior_ranges[0, 0] + (prior_ranges[0, 1] - prior_ranges[0, 0]) / (
                        1 + np.exp(-values_in_real[0]))
            return values_in_unif
        else:
            d = len(values_in_real)
            values_in_unif = [
                prior_ranges[i, 0] + (prior_ranges[i, 1] - prior_ranges[i, 0]) / (1 + np.exp(-values_in_real[i])) for i
                in range(d)]
            return np.array(values_in_unif)

class WFUniform():
    def __init__(self, neg_approx_llhd, population_size, generation, generation_interval,
                 recomb_param, haplotype_freq, data_obs, n_sample, n_param, n_ensemble = 10):
        ## neg_approx_llhd: the approximation of neglikelihood
        ## data_obs: observed data
        ## n_sample: number of particles in SMC would be used
        ## n_param = dimension of the parameter space
        self.core = ModelWrightFisher(neg_approx_llhd, n_param, population_size, generation, generation_interval,
                 recomb_param, haplotype_freq, rng=np.random.RandomState(1234))
        self.data_obs = data_obs
        self.n_sample = n_sample
        self.n_param = n_param
        self.n_ensemble = n_ensemble
        self.a = -.1
        self.b = .1
        ### Generate initail particles
        self.initial_params_original = torch.tensor(scp.stats.uniform(loc=self.a, scale=self.b-self.a).rvs(size=(self.n_sample,self.n_param)))
        self.initial_params = self.transformationtoR(self.initial_params_original)
        self.initial_weight = np.ones(self.n_sample) / self.n_sample

    def logprior_grad(self, thetaR, want_grad=False):
        ## Input thetaR lies in Real line, so transformation used to get to the correct parameter space ##
        theta, logJacobian = self.invtransformationfromR(thetaR, jacneeded=True)
        if want_grad:
            raise RuntimeError("Logprior pdf is not differentiable.")
        else:
            llhd = sum(scp.stats.uniform(loc=self.a, scale=(self.b-self.a)).logpdf(theta) + logJacobian)
            return llhd

    def llhd_grad(self, thetaR, want_grad=False):
        ## Input thetaR lies in Real line, so transformation used to get to the correct parameter space ##
        theta = self.invtransformationfromR(thetaR)
        if want_grad:
            raise RuntimeError("approx LogLHD of WF model is not differentiable.")
        else:
            llhd = self.core.approx_llhd(theta, self.data_obs, n_ensemble= self.n_ensemble, if_grad=want_grad)
            return llhd

    def invtransformationfromR(self, sampleR, jacneeded = False):
        logit_inv = (1/(1+torch.exp(-sampleR)))
        transformed = (self.b-self.a) * logit_inv + self.a
        logJacobian = np.log(self.b-self.a) + torch.log(logit_inv) + torch.log(1 - logit_inv)
        if jacneeded:
            return transformed, logJacobian.detach().tolist()
        else:
            return transformed

    def transformationtoR(self, sample):
        return torch.special.logit((1/(self.b-self.a)) * (sample - self.a))

class ModelWrightFisherHapFreq():
    ## Help file and description of the inputs ##
    def __init__(self, loss_fn, n_parameter, population_size, generation, generation_interval,
                 recomb_param, dominance_param = None, rng = None):
        self.loss_fn = loss_fn
        self.epistasis = False
        self.epistasis_alpha = 8
        self.L = n_parameter
        self.population_size = population_size
        self.generation = generation
        self.generation_interval = generation_interval
        if dominance_param is None:
            self.dominance_param = [.5 for ind in range(self.L)]
        else:
            self.dominance_param = dominance_param
        self.haplo = False
        self.recomb_param = recomb_param
        if rng is None:
            self.rng = np.random.RandomState(19)
        else:
            self.rng = rng

        ## Some partitions and their maps pre-calculated
        ## Calculate list of all possible non-empty partitions of {1,2,..,L}
        P_partitions = self.compute_IJ_partitions(r=self.recomb_param)
        self.P_partitions = P_partitions
        ## Now calculate i,j maps to i',j' for all partitions in P_partitions where i and j can be 1,..,2**L
        ij_map_Partition = np.zeros(shape=(2 ** self.L, 2 ** self.L, len(self.P_partitions), 2), dtype=int)
        for ind_iter_i in range(2 ** self.L):
            for ind_iter_j in range(2 ** self.L):
                for ind_P in range(len(self.P_partitions)):
                    ij_map_Partition[ind_iter_i, ind_iter_j, ind_P, 0] = \
                        int(self.get_haplotype_index(ind_iter_i, ind_iter_j, P_partitions[ind_P][0][0], self.L))
                    ij_map_Partition[ind_iter_i, ind_iter_j, ind_P, 1] = \
                        int(self.get_haplotype_index(ind_iter_j, ind_iter_i, P_partitions[ind_P][0][0], self.L))
        self.ij_map_Partition = ij_map_Partition


    def approx_llhd(self, parameters, data, n_ensemble = 10, if_grad = False):
        # Convert data to torch tensor
        #data = torch.Tensor(np.array(data))
        #!!!! This model can't be used with autograd!!!!!

        simulations = self.forward_simulation(parameters, n_data=n_ensemble)
        loss = -1 * self.loss_fn(simulations, data)
        if if_grad:
            raise RuntimeError("Loss is not differentiable for this model.")
        else:
            return loss.item()

    def forward_simulation(self, parameters, n_data=1):

        parameters_numpy = parameters.detach().numpy()
        selection_coeff = parameters_numpy[:self.L]
        # #### Initial Haplotype Frequency ####
        hapl_freq = np.array(parameters_numpy[self.L:])
        # #### Dominance parameter ####
        # h_all_z = parameters_numpy[self.L+pow(2, self.L)-1:2*self.L+pow(2, self.L)-1]
        # #### epistasis alpha parameter ####
        # epistasis_alpha_z = parameters_numpy[2*self.L+pow(2, self.L)-1:2*self.L+pow(2, self.L)]
        # ### Recombination Rate
        # r_all = parameters_numpy[2*self.L+pow(2, self.L):]
        # for every wf sim with fixed L I have the same fitness matrix W(Hij) and
        # same possible recombination events I, summarised into the list of P_partitions
        W_ij = self.compute_fitness_matrix_L_loci(s=selection_coeff, h=self.dominance_param, alpha=self.epistasis_alpha)
        self.W_ij = W_ij
        # however rho_ij(t) has to be updated within the simulator
        if len(self.generation_interval) == 1:
            timestep_generation = np.linspace(0, self.generation,
                                          int(self.generation / self.generation_interval[0]) + 1, dtype=int)
        else:
            timestep_generation = np.array([int(0)]+[int(x) for x in self.generation_interval])

        # Do the actual forward simulation
        vector_of_k_samples = self.wf_sim(fitness_matrix=W_ij,
                                          population_size=self.population_size,
                                          initial_hapl_freq=hapl_freq,
                                          last_gen=self.generation,
                                          timestep_generation=timestep_generation,
                                          num_forward_simulations=n_data, rng=np.random.RandomState(19))

        # # Do the actual forward simulation
        # vector_of_k_samples = self.wf_sim(fitness_matrix=W_ij,
        #                                   # P_partitions = P_partitions,
        #                                   population_size=self.population_size,
        #                                   initial_hapl_freq=hapl_freq,
        #                                   initial_gen=0,
        #                                   last_gen=self.generation,
        #                                   generation_interval=self.generation_interval,
        #                                   num_forward_simulations=n_data, rng=np.random.RandomState(self.seed))
        # Format the output to obey API
        return torch.stack(vector_of_k_samples, dim=1)

    # There are a series of functions that can be done once and used as defaults for
    # the rest of the simulations
    def compute_fitness_matrix_L_loci(self, s, h, alpha):
        """
        Computes the fitness matrix W(H_ij) with a general number of L loci.

        Args:
            s (list/numpy.ndarray (L,) ): list (L,) containing the selection coefficients.
            h (list/numpy.ndarray (L,) ): list (L,) containing the dominanc parameters.

        Returns:
            numpy.ndarray: A (2^L, 2^L) array containing the fitness values for pair of haplotypes.
        """
        if self.L == 1:
            W = np.array([[1, 1 + s[0] * h[0]],
                          [1 + s[0] * h[0], 1 + s[0]]])
        else:
            w_l_tensor = []
            for l in range(self.L):
                # create single (2x2) w_l matrices
                w_l_array = np.array([[1, 1 + s[l] * h[l]],
                                      [1 + s[l] * h[l], 1 + s[l]]])

                # Create a tuple containing the reshaping for matrix multiplication
                L_ones_l_list = [1] * self.L
                L_ones_l_list[l] = 2
                L_ones_l_tuple = tuple(L_ones_l_list + L_ones_l_list)
                w_l_array = w_l_array.reshape(L_ones_l_tuple)

                w_l_tensor.append(w_l_array)

            # matrix multiplication between w_l matrices
            W = 1
            for l in range(self.L):
                W = W * w_l_tensor[l]
            W = W.reshape(2 ** self.L, 2 ** self.L)

        if self.epistasis:
            if self.L == 1:
                Epis_X = np.array([[np.exp(0), np.exp(s * h)],
                                   [np.exp(s * h), np.exp(s)]])
            else:
                epis_X_l_tensor = []
                for l in range(self.L):
                    # create single (2x2) epis_X_l matrices
                    epis_X_l_array = np.array([[np.exp(0), np.exp(s[l] * h[l])],
                                               [np.exp(s[l] * h[l]), np.exp(s[l])]])

                    # Create a tuple containing the reshaping for matrix multiplication
                    L_ones_l_list = [1] * self.L
                    L_ones_l_list[l] = 2
                    L_ones_l_tuple = tuple(L_ones_l_list + L_ones_l_list)
                    epis_X_l_array = epis_X_l_array.reshape(L_ones_l_tuple)

                    epis_X_l_tensor.append(epis_X_l_array)

                # matrix multiplication between w_l matrices
                Epis_X = 1
                for l in range(self.L):
                    Epis_X = Epis_X * epis_X_l_tensor[l]
                Epis_X = alpha * np.power(np.log(Epis_X.reshape(2 ** self.L, 2 ** self.L)) / sum(s), 2)

        if self.epistasis:
            W = W * Epis_X

        return W


    def powerset(self, set):
        """
        Auxiliary function computing the powerset of a given set.

        Args:
            set (list): a non empty list of elements in the set.

        Returns:
            function to compute the powerset.

        e.g.
            powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        """
        s = list(set)  # allows duplicate elements
        return chain.from_iterable(combinations(s, r) for r in np.arange(1, len(s) + 1))


    def compute_IJ_partitions(self, r):
        """
        Computes the all possible pairs of recombination events IJ, partitioning
        the loci set into two exhaustive and non-empty sets for a general L-loci scenario.

        Args:
            r (list): A list of recombination rates for pairwise loci-break.

        Returns:
            A list of all partitions. Each partition is a tuple containing the two elements:
            the list of I, J sets (each I and J is a list with loci letters) and a list of
            one containing the conputed r_I coefficient.

        NOTE: The number of possible partitions, for L>1, can be computed exactly with
        the following formula: sum_{l = 1, L-1} binom(L-1, l)
        Code available:
        \{
            P_len = 0
            for l in range(L-1):
                P_len += scipy.special.binom(L-1, l)
            P_len = int(P_len)
        \}
        """
        L = self.L

        # call the letters I will use for the loci
        loci_alphabet = string.ascii_uppercase[:L]

        # create list of recombination rates with letter of the loci pair
        R_loci_values = []
        for l in range(L - 1):
            r_loci_value = [[loci_alphabet[l], loci_alphabet[l + 1]], [r[l]]]
            R_loci_values.append(r_loci_value)

        L_set = list(loci_alphabet)

        # Creating all possible Recombination events (I)
        L_set_noA = L_set.copy()
        L_set_noA.remove("A")
        all_other_partitions = self.powerset(L_set_noA)
        I_sets = [["A"]] + [["A"] + list(elem) for elem in all_other_partitions]
        I_sets = I_sets[:-1]

        P_len = len(I_sets)
        P_partitions = []
        for pp in range(P_len):
            current_J_set = L_set.copy()
            for e in range(len(I_sets[pp])):
                current_J_set.remove(I_sets[pp][e])

            p_part = [I_sets[pp], current_J_set]

            P_partitions.append(p_part)

        # Compute recomination rates according to events I
        for pp in range(P_len):
            # create an empty vector of length  = len(r) = L-1
            r_counts = np.zeros(L - 1, dtype=int)
            # for each event I, count how many elements of I each r coefficient involves (max 2)
            for e in range(len(I_sets[pp])):
                for r_rate in range(L - 1):
                    if I_sets[pp][e] in R_loci_values[r_rate][0]:
                        r_counts[r_rate] += 1
            # coefficient r enters the multiplication only if it includes 1 of the elements,
            # not if 0 or 2, so just set the 2s to 0.
            r_counts = [0 if rr == 2 else rr for rr in r_counts]
            if np.sum(r_counts) == 1:
                # only containing one r
                r_I = np.sum(np.array(r_counts) * r)
            else:
                # need to compute product of r coefficients
                r_I = np.prod(np.array(r_counts) * r)

            P_partitions[pp] = P_partitions[pp], [r_I]

        return P_partitions


    # end of preliminary functions

    def get_hapl_index_for_loci(self, L):
        """
        Auxiliary function producing a list of L lists,
        each indicating which indices of the hapl freq vector need to be considered
        to compute the freq of allele_1 of each locus.
        """

        length = 2 ** L
        sublists = [[] for _ in range(L)]

        for i in range(L):
            step = 2 ** (L - 1 - i)  # Number of elements to select and skip
            j = 0
            while j < length:
                # Select `step` elements
                sublists[i].extend(range(j, min(j + step, length)))
                # Skip `step` elements
                j += 2 * step
        return sublists


    def calculate_one_allele_freq(self, x_hapl_freq):
        """
        Takes a vector of haplotypes frequencies with L loci (each with two alleles)
        and returns the allele frequencies of type-1 allele (i.e.the wildtype).
        (For L = 1 there is a simpler one but I kept this general)
        """

        # we can deduce L by the lenght of the matrix
        L = int(np.log2(len(x_hapl_freq)))

        hapl_index_for_loci = self.get_hapl_index_for_loci(L)

        allele_frequencies = [np.sum(x_hapl_freq[index_locus_al]) for index_locus_al in hapl_index_for_loci]
        # I noticed a rounding error, with e-10 decimals
        return allele_frequencies


    def calculate_allele_freq_X(self, X_hapl_freq):
        """"
        Computes allele frequencies over multiple timepoints.
        """
        if len(X_hapl_freq.shape) == 1:
            ALL_allele_freq = self.calculate_one_allele_freq(X_hapl_freq)
        else:
            timepoints = X_hapl_freq.shape[1]

            ALL_allele_freq = []
            for tt in range(timepoints):
                # print(self.calculate_one_allele_freq(X_hapl_freq[:,tt]))
                ALL_allele_freq.append(self.calculate_one_allele_freq(X_hapl_freq[:, tt]))
        # print(ALL_allele_freq)
        return np.vstack(ALL_allele_freq)


    def wf_sim(self, fitness_matrix,
               population_size, initial_hapl_freq, last_gen, timestep_generation, num_forward_simulations, rng):
        """
        Simulate n replicates of a discrete wf trajectory.

        Returns n matrices of shape (t, L) for the allele frequencies at each timepoint,
        Here, rng is passing the fixed random state to each of the simulations.
        """
        # define equally spaced discrete timepoints for the generations
        timestep_generation_normalized = timestep_generation / max(timestep_generation)

        # print(timestep_generation)

        result_allele = []
        result_haplo = []

        for iter_simul in range(num_forward_simulations):
            # if num_forward_simulations>1:
            #    current_rng_r = rng_rep[n]
            # else:
            haplo_freq_trajectory = self.compute_haplo_freq_traj(w=fitness_matrix,
                                                                 x_initial=initial_hapl_freq,
                                                                 gen=last_gen,
                                                                 population_size=population_size,
                                                                 rng=rng)[timestep_generation, :]
            # haplo
            result_haplo.append(haplo_freq_trajectory.T)

            # allele
            allele_freq_trajectory = self.calculate_allele_freq_X(haplo_freq_trajectory.T)
            #print(allele_freq_trajectory.shape)
            #print(np.hstack((timestep_generation.reshape(-1,1), allele_freq_trajectory)).shape)
            allele_freq_trajectory = torch.tensor(np.hstack((timestep_generation_normalized.reshape(-1,1), allele_freq_trajectory)))
            #allele_freq_trajectory = torch.tensor(allele_freq_trajectory)
            #print(allele_freq_trajectory.shape)
            #allele_freq_trajectory = allele_freq_trajectory.T.reshape(-1)
            # print(allele_freq_trajectory.shape)
            # allele_freq_trajectory = np.insert(allele_freq_trajectory,
            #                                    0, len(timestep_generation))
            # print(allele_freq_trajectory.shape)
            result_allele.append(allele_freq_trajectory)


        #result_haplo = [np.array([x]).reshape(-1, ) for x in result_haplo]
        #result_haplo = [np.insert(np.around(k, 3), 0, int(last_gen / generation_interval) + 1) for k in result_haplo]

        #if self.haplo:
        #    return result_haplo, result_allele
        #else:
        return result_allele

    def compute_haplo_freq_traj(self, w, x_initial, rng, gen=60, population_size=1000):
        """
        Calculates the haplotype frequencies over time under genetic drift given
        the haplotype frequencies tilde{x}_i after accounting for recombination
        and selection.
        This function is independent of the number of loci, L.

        Args:
            x (numpy.ndarray): An array containing the starting frequencies of the haplotypes.

        Returns:
            numpy.ndarray: A 2D array with dimensions (t, 8) representing the haplotype frequencies at each timepoint.
        """
        freq = np.zeros((gen + 1, 2 ** self.L))
        freq[0, :] = x_initial
        x = x_initial
        for iter_gen in range(gen):
            RHO_t = self.compute_rho_ij(x, self.P_partitions, self.L)
            x_tilde = self.compute_x_tilde_hapl_freq(w=w,x=x,rho=RHO_t)
            # already normalised (but just in case)
            # x_tilde = x_tilde/np.sum(x_tilde)
            x = rng.multinomial(2 * population_size, x_tilde) / (2 * population_size)
            freq[iter_gen + 1, :] = x
        # print(freq.shape)
        return freq


    def index_to_haplotype(self, index, L):
        """
        Given an index, this function converts it into the corresponding
        haplotype for a general number of loci.
        The index ranges from 0 to 2^L, and the haplotype ranges from [0]*L to [1]*L.


        Args:
            index (int): index of the current haplotype.
            L (int): number of loci.
        Returns:
            tuple: A (L,) tuple with 0-1s with each zeros indicating allele for
            each corresponding positional locus.
        """

        if index >= 2 ** L:
            raise RuntimeError(
                'Index cannot be larger than max(2^L-1).')

        # Write it in a general way
        haplotype_list = [index // (2 ** (L - 1))]
        for i in np.arange(1, L - 1):
            current_locus_index = (index % 2 ** (L - i)) // (2 ** (L - i - 1))
            haplotype_list.append(current_locus_index)
        # Last one is fixed
        haplotype_list.append(index % 2)

        return tuple(haplotype_list)


    def get_haplotype_index(self, h1, h2, I, L):
        """
        This function takes two haplotype indices (h1 and h2) and a set I (decomposition of L).
        It returns the index of the haplotype formed by combining components of h1 and h2
        based on the decomposition I.
        It works with L>1 as assessed in higher function.

        Args:
            h1 (int): index of haplotype 1.
            h2 (int): index of haplotype 2.
            I (list): list with the letters of loci contained in the I decomposition.
            L (int): number of loci.
        Returns:
            int: th index of the resulting haplotype
            from the recombination of h1 and hy according to I.

        NOTE: the output using (h1, h2, I, L) is the same as using (h1, h2, J, L),
        with I-J coming from same partition.
        """

        # Convert the indices h1 and h2 to their corresponding haplotypes.
        h1_haplotype = self.index_to_haplotype(h1, L)
        h2_haplotype = self.index_to_haplotype(h2, L)

        # Create the resulting haplotype by choosing components from h1_haplotype if the
        # index is in I, and from h2_haplotype if the index is not in I.
        # hence: H_(i_I, j_J)
        upper_alphabet = list(string.ascii_uppercase[:L])
        # technically we know hoe to obtain H_(j_I, i_J) in an analogous way

        resulting_hapl = [h1_haplotype[ii] if locus_l in I else h2_haplotype[ii] for ii, locus_l in
                          enumerate(upper_alphabet)]

        # Convert the resulting haplotype back to an index, which is returned.
        resulting_index = 0
        for l in range(L):
            resulting_index = resulting_index + resulting_hapl[l] * (2 ** (L - (l + 1)))

        return resulting_index


    def compute_rho_ij(self, x, P_partitions, L):
        """
        Computes the recombination matrix RHO_ij(t).
        Works with L>1 as specified in upper function.

        Args:
            x (numpy.ndarray): An array containing the haplotype frequencies x(t).
            P_partitions (list): list of all possible recombination events
            L (int): number of loci.
        Returns:
            numpy.ndarray: A (2^l, 2^L) array representing the recombination term
            for each haplotype.

        NOTE: the returning RHO_ij is a squared symmetric matrix with 0's in the main diagonal.
        """
        # Initialize an array of (2^L)x(2^L) for all rho's
        rho_ij = np.zeros((2 ** L, 2 ** L))

        if L > 1:
            # If L = 1, all rho_ij's stay zero.

            # Iterate over all couplings i-j
            for i in range(2 ** L):
                for j in np.arange(i + 1, 2 ** L):
                    if len(P_partitions) > 1:  # L>2
                        event_I_to_rho_ij = np.zeros(len(P_partitions))
                        for pp in range(len(P_partitions)):
                            #current_I = P_partitions[pp][0][0]  # select the IJ partition, specifically I
                            current_r_I = P_partitions[pp][1][0]  # select r_I

                            # Get the indices of the haplotypes formed by combining components
                            # # of haplotypes i and j based on the decomposition I.
                            #i_Ij_J_index = self.get_haplotype_index(i, j, current_I, L)
                            #j_Ii_J_index = self.get_haplotype_index(j, i, current_I, L)

                            i_Ij_J_index = self.ij_map_Partition[i,j,pp,0]
                            j_Ii_J_index = self.ij_map_Partition[i,j,pp,1]

                            # contribution of current I
                            event_I_to_rho_ij[pp] = current_r_I * (x[i] * x[j] - x[i_Ij_J_index] * x[j_Ii_J_index])
                        rho_ij[i, j] = np.sum(event_I_to_rho_ij)
                        rho_ij[j, i] = rho_ij[i, j]

                    # When you just have two loci, L = 2
                    else:
                        #current_I = P_partitions[0][0][0]  # select the IJ part, and I
                        current_r_I = P_partitions[0][1][0]  # select r_I

                        # consider both I-J swaps
                        #i_Ij_J_index = self.get_haplotype_index(i, j, current_I, L)
                        #j_Ii_J_index = self.get_haplotype_index(j, i, current_I, L)
                        i_Ij_J_index = self.ij_map_Partition[i, j, 0, 0]
                        j_Ii_J_index = self.ij_map_Partition[i, j, 0, 1]

                        # contribution of current I
                        rho_ij[i, j] = current_r_I * (x[i] * x[j] - x[i_Ij_J_index] * x[j_Ii_J_index])
                        rho_ij[j, i] = current_r_I * (x[i] * x[j] - x[i_Ij_J_index] * x[j_Ii_J_index])
        return rho_ij


    def compute_x_tilde_hapl_freq(self, w, x, rho):
        """
        Computes the next generation haplotyple frequencies tilde{x}_(t+1)
        accounting for recombination and selection.
        """

        # could be more efficient to include it up there as it would be the same i - j loop
        # as r_ij
        x_tilde = np.zeros(x.shape)
        for i in range(len(x_tilde)):
            unnormalised_x_tilde_i = np.zeros(x.shape)
            for j in range(len(x_tilde)):
                unnormalised_x_tilde_i[j] = w[i, j] * (x[i] * x[j] - rho[i, j])
            x_tilde[i] = np.sum(unnormalised_x_tilde_i)
        # Normalize the recombination/selection haplotype frequencies x_tilde_i by
        # the sum of the outer product of the haplotype frequencies multiplied by the fitness matrix.

        common_denom = np.sum(w * np.outer(x, x))
        x_tilde = x_tilde / common_denom
        return x_tilde


    def transform_Rd_to_Simplex(self, We):
        """
        This function applies a reverse transformation to the realisation of a Dirichlet sample in Rd
        back to the original sample space.

        In our context:
        Computes the initial haplotype frequencies (x_0 in the 2^L simplex)
        using 2^L -1 dimensional random variable in Rd, denoted as -We- \in R 2^{L}-1
            Args:
                We:         2^{L}-1 realisations from Dir random variables in real line (-inf, +inf)
            Returns:
                x_cand_0:   The candidate vector of initial haplotype frequencies x_0 in R 2^{L}
        """

        We = np.array(We)
        m = len(We) + 1
        Z_vector = np.zeros(m - 1)
        x_dir = np.zeros(m)
        for k in np.arange(1, m):
            # logit 1/(m-k+1)
            p_mk1 = 1 / (m - k + 1)
            z_arg = We[k - 1] + np.log(p_mk1 / (1 - p_mk1))
            # inv logit
            Z_vector[k - 1] = 1 / (1 + np.exp(-z_arg))

            if k == 1:
                x_dir[k - 1] = Z_vector[k - 1]
            else:
                x_dir[k - 1] = (1 - np.sum(x_dir[:k])) * Z_vector[k - 1]

        # k = m
        x_dir[-1] = 1 - np.sum(x_dir[:-1])

        return x_dir


    def transform_Rd_to_Constrained(self, values_in_real, prior_ranges):
        """
        This function applies an inverse transformation to the realisation of d-dimensional samples
        in Rd back to the uniform range.

        Transform back samples from the Rd into d-uniform with their own ranges.
            Args:
                values_in_real (d-dim array):       Realisation from d uniform distributions in real line range (R^d).
                                                    supports atomic values
                prior_ranges (dx2 array):           Array of prior ranges of the uniforms.
            Returns:
                values_in_unif:                     The output vector of values in d-uniforms
        """
        if isinstance(values_in_real, int):
            values_in_unif = prior_ranges[0] + (prior_ranges[1] - prior_ranges[0]) / (1 + np.exp(-values_in_real))
            return list(values_in_unif)[0]
        elif len(values_in_real) == 1:
            values_in_unif = prior_ranges[0, 0] + (prior_ranges[0, 1] - prior_ranges[0, 0]) / (
                        1 + np.exp(-values_in_real[0]))
            return values_in_unif
        else:
            d = len(values_in_real)
            values_in_unif = [
                prior_ranges[i, 0] + (prior_ranges[i, 1] - prior_ranges[i, 0]) / (1 + np.exp(-values_in_real[i])) for i
                in range(d)]
            return np.array(values_in_unif)

class WFUniformDirichlet():
    def __init__(self, neg_approx_llhd, population_size, generation, generation_interval,
                 recomb_param, data_obs, n_sample, n_param, dominance_param = None):
        ## neg_approx_llhd: the approximation of neglikelihood
        ## data_obs: observed data
        ## n_sample: number of particles in SMC would be used
        ## n_param = dimension of the parameter space
        if dominance_param is None:
            dominance_param = [.5 for ind in range(n_param)]
        self.core = ModelWrightFisherHapFreq(neg_approx_llhd, n_param, population_size, generation, generation_interval,
                 recomb_param, dominance_param=dominance_param)
        self.data_obs = data_obs
        self.n_sample = n_sample
        ### The following self.n_param is the dimension of real valued space on which SMC happens
        self.n_param = n_param + pow(2, n_param) - 1
        ### Bounds for Uniform prior
        self.a, self.b = -1.0, 1.0
        ### Generate initail particles
        self.initial_params_original_selcof = torch.tensor(scp.stats.uniform(loc=self.a, scale=(self.b-self.a)).rvs(size=(self.n_sample,self.core.L)))
        self.dirich = Dirichlet((1/pow(2, self.core.L)) * torch.ones(pow(2, self.core.L)))
        self.initial_params_original_hapfre = self.dirich.sample((self.n_sample,1)).squeeze()
        self.initial_params_original = torch.cat((self.initial_params_original_selcof, self.initial_params_original_hapfre),dim=1)
        self.initial_params = torch.zeros(size=(self.initial_params_original.shape[0], self.initial_params_original.shape[1]-1))
        for iter in range(self.initial_params_original.shape[0]):
            self.initial_params[iter,:] = self.transformationtoR(self.initial_params_original[iter,:])
        self.initial_weight = np.ones(self.n_sample) / self.n_sample

    def logprior_grad(self, thetaR, want_grad=False):
        ## Input thetaR lies in Real line, so transformation used to get to the correct parameter space ##
        funkeeprior = lambda x: self.logprior_local(x)
        if want_grad:
            #raise RuntimeError("Logprior pdf is not differentiable.")
            return funkeeprior(thetaR), self.zeroth_order_grad(func=funkeeprior, x=thetaR)
        else:
            return funkeeprior(thetaR)

    def logprior_local(self, thetaR, want_grad = False):
        theta, logJacobian = self.invtransformationfromR(thetaR, jacneeded = True)
        llhd = sum(scp.stats.uniform(loc=self.a, scale=(self.b-self.a)).logpdf(theta[:self.core.L]))\
               + self.dirich.log_prob(theta[self.core.L:]).item() + logJacobian
        return llhd

    def llhd_grad(self, thetaR, want_grad=False):
        ## Input thetaR lies in Real line, so transformation used to get to the correct parameter space ##
        #theta = self.invtransformationfromR(thetaR)
        funkee = lambda x: self.llhd_local(x)
        if want_grad:
            #raise RuntimeError("approx LogLHD of WF model is not differentiable.")
            # We compute using zeroth order gradient
            return funkee(thetaR), self.zeroth_order_grad(func=funkee, x=thetaR)
        else:
            return funkee(thetaR)

    def llhd_local(self, thetaR, want_grad = False):
        theta = self.invtransformationfromR(thetaR)
        llhd = self.core.approx_llhd(theta, self.data_obs, if_grad=want_grad)
        return llhd

    def zeroth_order_grad(self, func, x, mu=0.01, b=30):
        '''
        Here mu is a smoothing parameter and b is the number of random directions to sample
        Try with small values of mu like 0.5, 0.1, 0.01 etc.
        You can take b = 10/ 20/ 30 etc. Bigger value of b will give better approximation of the gradient but will be slower.
        '''

        n_param_zero = x.shape[0]
        f_x = func(x)
        grad = np.zeros_like(x)
        for i in range(b):
            random_u = multivariate_normal.rvs(mean=np.zeros(n_param_zero), cov=np.eye(n_param_zero), size=1)
            new_x = x + random_u * mu
            f_xplusu = func(new_x)
            new_grad = ((f_xplusu - f_x) * random_u)
            #print(grad, new_grad)
            grad = grad + new_grad / (mu * b)
        return torch.tensor(grad)

    def invtransformationfromR(self, sampleR, jacneeded = False):
        # first_part
        logit_inv = (1/(1+torch.exp(-sampleR[:self.core.L])))
        first_part_transformed = (self.b-self.a) * logit_inv + self.a
        first_part_logJacobian = np.log(self.b - self.a) + torch.log(logit_inv) + torch.log(1 - logit_inv)
        # second_part
        transformed, second_part_logJacobian = self.transform_Rd_to_Simplex_Jacobian(sampleR[self.core.L:].detach().numpy())
        second_part_transformed = torch.tensor(transformed)
        # final Jacobian
        Jacobian = sum(first_part_logJacobian) + second_part_logJacobian
        if jacneeded:
            return torch.cat((first_part_transformed, second_part_transformed)), Jacobian.detach().numpy()
        else:
            return torch.cat((first_part_transformed, second_part_transformed))

    def transformationtoR(self, sample):
        first_part = torch.special.logit((1/(self.b-self.a)) * (sample[:self.core.L] - self.a))
        second_part = torch.tensor(self.transform_Simplex_to_Rd(sample[self.core.L:].detach().numpy()))
        return torch.cat((first_part,second_part))

    # Transformations for Dirichlet priors and the Real line
    def transform_Simplex_to_Rd(self, w):
        """
        This function applies a transformation to the realisation of a Dirichlet sample into Rd.

        Transform dirichlet realizations (x_0 in the 2^{L} simplex) into a R 2^{L}-1 dimensional random
        vector in the real starting from a 2^L -1 dimensional random variable in Rd,
        denoted as -We- \in R 2^{L}-1
            Args:
                x_haplo:     Realisation from a dirichlet in the 2^{L} simplex
            Returns:
                w_Rd:       The output vector of initial haplotype frequencies in the R^ 2^{L}-1, (-inf, +inf)
        """

        if isinstance(w, list):
            w = np.array(w)

        W = np.zeros(len(w)-1)
        for iter in range(len(w)-1):
            W[iter] = scp.special.logit(w[iter] / sum(w[iter:])) - scp.special.logit(1/(len(w)-(iter+1)+1))
        return W

    def transform_Rd_to_Simplex_Jacobian(self, W):
        """
        This function applies a reverse transformation to the realisation of a Dirichlet sample in Rd
        back to the original sample space.

        In our context:
        Computes the initial haplotype frequencies (x_0 in the 2^L simplex)
        using 2^L -1 dimensional random variable in Rd, denoted as -We- \in R 2^{L}-1
            Args:
                We:         2^{L}-1 realisations from Dir random variables in real line (-inf, +inf)
            Returns:
                x_cand_0:   The candidate vector of initial haplotype frequencies x_0 in R 2^{L}
        """

        W = np.array(W)

        z, w, JacTerm = np.zeros(len(W)), np.zeros(len(W)+1), np.zeros(len(W))

        for iter in range(len(z)):
            z[iter] = scp.special.expit(W[iter] + scp.special.logit(1/(len(w)-(iter+1)+1)))

        w[0] = z[0]
        for iter in range(1, len(z)):
            w[iter] = z[iter] * (1 - sum(w[:iter]))

        w[-1] = 1 - sum(w)

        ## Calculation of Jacobian
        logJacobian = 0
        for iter in range(1, len(z)):
            logJacobian = np.log(z[iter]) + np.log(1 - z[iter]) + np.log((1 - np.sum(w[:iter-1])))

        return w, logJacobian

class WFUniformDirichlet2():
    def __init__(self, neg_approx_llhd, population_size, generation, generation_interval,
                 recomb_param, data_obs, n_sample, n_param, dominance_param = None):
        ## neg_approx_llhd: the approximation of neglikelihood
        ## data_obs: observed data
        ## n_sample: number of particles in SMC would be used
        ## n_param = dimension of the parameter space
        if dominance_param is None:
            dominance_param = [.5 for ind in range(n_param)]
        self.core = ModelWrightFisherHapFreq(neg_approx_llhd, n_param, population_size, generation, generation_interval,
                 recomb_param, dominance_param=dominance_param)
        self.data_obs = data_obs
        self.n_sample = n_sample
        ### The following self.n_param is the dimension of real valued space on which SMC happens
        self.n_param = n_param + pow(2, n_param) - 1
        ### Bounds for Uniform prior
        self.a, self.b = -0.1, 0.1
        ## Simplex R transformer
        self.STFR = SimplexToFromRd()
        ### Generate initail particles
        self.initial_params_original_selcof = torch.tensor(scp.stats.uniform(loc=self.a, scale=(self.b-self.a)).rvs(size=(self.n_sample,self.core.L)))
        self.dirich = Dirichlet((1/pow(2, self.core.L)) * torch.ones(pow(2, self.core.L)))
        self.initial_params_original_hapfre = self.dirich.sample((self.n_sample,1)).squeeze()
        self.initial_params_original = torch.cat((self.initial_params_original_selcof, self.initial_params_original_hapfre),dim=1)
        self.initial_params = torch.zeros(size=(self.initial_params_original.shape[0], self.initial_params_original.shape[1]))
        for iter in range(self.initial_params_original.shape[0]):
            self.initial_params[iter,:] = self.transformationtoR(self.initial_params_original[iter,:])
        self.initial_weight = np.ones(self.n_sample) / self.n_sample

    def logprior_grad(self, thetaR, want_grad=False):
        ## Input thetaR lies in Real line, so transformation used to get to the correct parameter space ##
        funkeeprior = lambda x: self.logprior_local(x)
        if want_grad:
            #raise RuntimeError("Logprior pdf is not differentiable.")
            return funkeeprior(thetaR), self.zeroth_order_grad(func=funkeeprior, x=thetaR)
        else:
            return funkeeprior(thetaR)

    def logprior_local(self, thetaR, want_grad = False):
        theta, logJacobian = self.invtransformationfromR(thetaR, jacneeded = True)
        llhd = sum(scp.stats.uniform(loc=self.a, scale=(self.b-self.a)).logpdf(theta[:self.core.L]))\
               + self.dirich.log_prob(theta[self.core.L:]).item() + logJacobian
        return llhd

    def llhd_grad(self, thetaR, want_grad=False):
        ## Input thetaR lies in Real line, so transformation used to get to the correct parameter space ##
        #theta = self.invtransformationfromR(thetaR)
        funkee = lambda x: self.llhd_local(x)
        if want_grad:
            #raise RuntimeError("approx LogLHD of WF model is not differentiable.")
            # We compute using zeroth order gradient
            return funkee(thetaR), self.zeroth_order_grad(func=funkee, x=thetaR)
        else:
            return funkee(thetaR)

    def llhd_local(self, thetaR, want_grad = False):
        theta = self.invtransformationfromR(thetaR)
        llhd = self.core.approx_llhd(theta, self.data_obs, if_grad=want_grad)
        return llhd

    def zeroth_order_grad(self, func, x, mu=0.01, b=30):
        '''
        Here mu is a smoothing parameter and b is the number of random directions to sample
        Try with small values of mu like 0.5, 0.1, 0.01 etc.
        You can take b = 10/ 20/ 30 etc. Bigger value of b will give better approximation of the gradient but will be slower.
        '''

        n_param_zero = x.shape[0]
        f_x = func(x)
        grad = np.zeros_like(x)
        for i in range(b):
            random_u = multivariate_normal.rvs(mean=np.zeros(n_param_zero), cov=np.eye(n_param_zero), size=1)
            new_x = x + random_u * mu
            f_xplusu = func(new_x)
            new_grad = ((f_xplusu - f_x) * random_u)
            #print(grad, new_grad)
            grad = grad + new_grad / (mu * b)
        return torch.tensor(grad)

    def invtransformationfromR(self, sampleR, jacneeded = False):
        # first_part
        logit_inv = (1/(1+torch.exp(-sampleR[:self.core.L])))
        first_part_transformed = (self.b-self.a) * logit_inv + self.a
        first_part_logJacobian = np.log(self.b - self.a) + torch.log(logit_inv) + torch.log(1 - logit_inv)
        # second_part
        #transformed, second_part_logJacobian = self.transform_Rd_to_Simplex_Jacobian(sampleR[self.core.L:].detach().numpy())
        transformed, second_part_logJacobian = self.STFR.simplexfromR(sampleR[self.core.L:].detach().numpy())
        second_part_transformed = torch.tensor(transformed)
        # final Jacobian
        Jacobian = sum(first_part_logJacobian) + second_part_logJacobian
        if jacneeded:
            return torch.cat((first_part_transformed, second_part_transformed)), Jacobian.detach().numpy()
        else:
            return torch.cat((first_part_transformed, second_part_transformed))

    def transformationtoR(self, sample):
        first_part = torch.special.logit((1/(self.b-self.a)) * (sample[:self.core.L] - self.a))
        #second_part = torch.tensor(self.transform_Simplex_to_Rd(sample[self.core.L:].detach().numpy()))
        second_part = torch.tensor(self.STFR.simplextoR(sample[self.core.L:].detach().numpy()))
        return torch.cat((first_part,second_part))
