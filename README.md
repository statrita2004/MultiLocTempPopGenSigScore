This is the repository for the article :
[Signature-Informed Selection Detection: A Novel Method for Multi-Locus Temporal Population Genetic Model with Recombination](https://doi.org/10.48550/arXiv.2512.14353)

- Codes were run and tested using Python 3.9.6
- **requirements.txt**: All the Python packages needed to run codes of this repository and can be installed using 'pip'
- **[sigkernel](https://github.com/crispitagorico/sigkernel)** repository is needed in addition and can be installed using 'pip install git+https://github.com/crispitagorico/sigkernel.git'

- **template.ipynb**: A python notebook illustrating how the inference was run for one replicate of Drosophila dataset, this could be used with modifications for the inference of your own dataset. 

## mysrc: A folder containing all the source code
- **ModelWF.py**: Code implementing Wright Fisher dynamics model
- **ModelFDS.py**: Code implementing Frequency Dependent Selection Wright Fisher dynamics model
- **scoring_rules.py**: Code implementing scoring rules including signature kernel score used here
- **MH.py**: Code implementing Metropolis-Hastings algorithm to sample from the scoring rule posterior
- **utils.py**: Code implementing some additional utilities necessary

## Codes replicating experiments in the article
- **Popgen2.py**: Simulates data from 2-loci WF model and draws samples from the scoring rule posterior of selection coefficients, finally produces Figure 1
- **PopGen2Estimate.py**: Studies behaviour of posterior mean of scoring rule posterior as estimate of selection coefficients for 2-loci WF model, producing Table 1
- **Popgen3.py**: Simulates data from 3-loci WF model and draws samples from the scoring rule posterior of selection coefficients, finally produces Figure 2
- **PopGen3Estimate.py**: Studies behaviour of posterior mean of scoring rule posterior as estimate of selection coefficients for 3-loci WF model, producing Table 2
- **PopGen2EstimatePerturbedSamplingError.py**: Studies behaviour of posterior mean of scoring rule posterior as estimate of selection coefficients for 2-loci WF model under sampling error with changing sequencing coverage, producing Table 3
- **PopGen2EstimateNeError.py**: Studies behaviour of posterior mean of scoring rule posterior as estimate of selection coefficients for 2-loci WF model under unknown demography, producing Table 4
- **PopGen1FDS.py**: Simulates data from 1-loci FDS model and draws samples from the scoring rule posterior of selection coefficients, finally produces Figure 3a
- **PopGen2FDS.py**: Simulates data from 2-loci FDS model and draws samples from the scoring rule posterior of selection coefficients, finally produces Figure 3b
- **PopGen3FDS.py**: Simulates data from 3-loci FDS model and draws samples from the scoring rule posterior of selection coefficients, finally produces Figure 3c-d
- **YeastWF.py**: Performs posterior inference for selection coefficients for Yeast dataset, Figure 4-5 and Table 5
- **Popgen2hapfreq.py**: Simulates data from 2-loci WF model and draws samples from the joint scoring rule posterior of selection coefficients and initial haplotype frequency, finally produces Figure 6
- **DrosophilaWFhapfreq.py**: Performs joint posterior inference for selection coefficients and initial haplotype frequency for Drosophila dataset, finally producing Figure 7, Figure 10-18 (in Appendix) and Table 6

## Data: A folder containing the real experimental datasets used in this article

- **Yeast**: The experimental dataset of Yeast [CITE] used in this article. 
- **Drosophila**: The experimental dataset of Drosophila [CITE] used in this article.

## Results: A folder containing all the inferentential results and figures reported in this article

- **WF**: Results related to the experiments of Wright Fisher model
- **FDS**: Results related to the experiments of Frequency dependent selection model
- **Yeast**: Results related to the real data analysis of Yeast
- **Drosophila**: Results related to the real data analysis of Drosophila

## backend: A folder containing codes to efficiently parallelise python codes using MPI
