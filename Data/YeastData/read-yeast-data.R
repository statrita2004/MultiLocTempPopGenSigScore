


#################################################################
# R functions for data extraction
# for actual extraction see below
#################################################################
split_freqs_column <- function(df, col_name,new_col_names) {
  # this function takes the column with the haplotype frequencies and 
  # puts it into a separate numeric columns named with 
  # user specified new name
  # Ensure input column is character (not factor)
  df[[col_name]] <- as.character(df[[col_name]])
  
  # Split the column by ";"
  split_matrix <- do.call(rbind, strsplit(df[[col_name]], split = ";"))
  split_matrix <- apply(split_matrix, 2, as.numeric)
  # Create new column names
  new_col_names <- paste0(new_col_names, "_", seq_len(ncol(split_matrix)))
  
  # Combine original df with new columns
  df_out <- cbind(df, split_matrix)
  
  # Rename the new columns
  colnames(df_out)[(ncol(df) + 1):ncol(df_out)] <- new_col_names
  
  return(df_out)
}


ds0 <- split_freqs_column(read.table("8s_haps_0.txt",header=T),"founderfreqs",
                          "hapfreq")

ds6 <- split_freqs_column(read.table("8s_haps_6.txt",header=T),"founderfreqs",
                          "hapfreq") 

ds12 <- split_freqs_column(read.table("8s_haps_12.txt",header=T),"founderfreqs",
                          "hapfreq") 

dk0 <- split_freqs_column(read.table("8k_haps_0.txt",header=T),"founderfreqs",
                          "hapfreq")

dk6 <- split_freqs_column(read.table("8k_haps_6.txt",header=T),"founderfreqs",
                          "hapfreq") 

dk12 <- split_freqs_column(read.table("8k_haps_12.txt",header=T),"founderfreqs",
                           "hapfreq") 

# check where the haplotype frequencies 
# change a lot

inverse_multinomial_vcov <- function(p) {
  # p: probability vector summing to 1
  
  # Remove last category to get (k-1) vector
  p_sub <- p[-length(p)]
  
  # Compute inverse covariance matrix of size (k-1) x (k-1)
  inv_cov <- diag(1 / p_sub) + 1 / p[length(p)]
  
  return(inv_cov)
}

mahalanobis_multinomial <- function(p, q) {
  # p, q: probability vectors summing to 1 and of same length
  
  # Check inputs
  if (length(p) != length(q)) stop("Vectors p and q must be same length")
  if (abs(sum(p) - 1) > 1e-2 || abs(sum(q) - 1) > 1e-2) stop("Vectors must sum to 1")
  
  # Remove last category
  p_sub <- p[-length(p)]
  q_sub <- q[-length(q)]
  
  # Compute inverse covariance matrix based on p
  inv_cov <- inverse_multinomial_vcov((p+q)/2)
  
  # Calculate difference vector
  diff_vec <- q_sub - p_sub
  
  # Compute Mahalanobis distance
  dist <- sqrt(t(diff_vec) %*% inv_cov %*% diff_vec)
  
  return(as.numeric(dist))
}

# Function to apply row-wise
mahalanobis_multinomial_rows <- function(A, B) {
  if (!all(dim(A) == dim(B))) stop("A and B must have the same dimensions")
  
  n <- nrow(A)
  dists <- numeric(n)
  
  for (i in 1:n) {
    p <- A[i, ]
    q <- B[i, ]
    
    # Basic checks (optional)
    if (abs(sum(p) - 1) > 1e-2 || abs(sum(q) - 1) > 1e-2) {
      stop(paste("Row", i, "does not sum to 1"))
    }
    
    dists[i] <- mahalanobis_multinomial(p, q)
  }
  
  return(dists)
}

#################################################################
# data extraction
#################################################################

setwd('/Users/af/pCloud Drive/m-pCloud/aktuell/Rito/Daten/yeast')

## data extraction: s and k code the replicate population, 
#  0, 6 and 12 are the cycle numbers

ds0 <- split_freqs_column(read.table("8s_haps_0.txt",header=T),"founderfreqs",
                          "hapfreq")

ds6 <- split_freqs_column(read.table("8s_haps_6.txt",header=T),"founderfreqs",
                          "hapfreq") 

ds12 <- split_freqs_column(read.table("8s_haps_12.txt",header=T),"founderfreqs",
                           "hapfreq") 

dk0 <- split_freqs_column(read.table("8k_haps_0.txt",header=T),"founderfreqs",
                          "hapfreq")

dk6 <- split_freqs_column(read.table("8k_haps_6.txt",header=T),"founderfreqs",
                          "hapfreq") 

dk12 <- split_freqs_column(read.table("8k_haps_12.txt",header=T),"founderfreqs",
                           "hapfreq")

# find positions with both a clear selection signal and consistent behavior 
# across both replicate popullation
# if there are multiple slectied haplotypes, it could be that due to 
# chance, one responds (changes frequency) a lot in one population, #
# and another one in the other replicate
# so one could also pick regions with inconsistent signals if desired

  mk0<- as.matrix(dk0[,10:17])
  mk6<- as.matrix(dk6[,10:17])
  mk12<- as.matrix(dk12[,10:17])
  ms0<- as.matrix(ds0[,10:17]) 
  ms6<- as.matrix(ds6[,10:17]) 
  ms12<- as.matrix(ds12[,10:17])
  
  mmk <-mahalanobis_multinomial_rows(mk0,mk12)
  mms <-mahalanobis_multinomial_rows(ms0,ms12)
  qk90<-quantile(mmk,0.9)
  qs90<-quantile(mms,0.9,na.rm=T)

  idx<-c(1:length(mmk))[(mmk>qk90)&(mms>qs90)]
  
  diag((mk0[idx,]-mk12[idx,])%*%t(ms0[idx,]-ms12[idx,]))
  
  # we choose window 7914
  final.hap.data<- rbind(mk0[7914,],mk6[7914,],mk12[7914,],ms0[7914,],ms6[7914,],ms12[7914,])
  rownames(final.hap.data)<-c("k-0","k-6","k-12","s-0","s_6","s-12")
  write.table(final.hap.data,"haplotype-frequencies.txt")
  
  
              # chromosome and starting position of chosen window
  c(dk0[7914,]$chr,dk0[7914,]$pos,dk0[7915,]$pos-1)
  
  
  # now choose some SNPs from the chosen window
  
  ### SNP data
  
  SNPS<-read.table("Inputfile_8.txt",header=T)
  SNPSsel=SNPS[(SNPS$CHROM=="chr4")&(SNPS$POS>=690800)&(SNPS$POS<693800),]
  
  # select 3 SNPs:   
  SNPSsel=SNPSsel[SNPSsel$POS %in% c(690968,692033,693728),]
  
  SNPSsel=SNPSsel[,c(1:4,c(21,23,25,27,29,31,18,24,26,28,30,32))]  



  SNP_dat = t(SNPSsel)  
  new_names <- paste(SNP_dat[1, ], SNP_dat[2, ], sep = "_")
  colnames(SNP_dat) <- new_names
  # Step 2: Remove the first two rows
  SNP_dat <- SNP_dat[-c(1:4), ] 
 SNP_freq = cbind(SNP_dat[1:6,],SNP_dat[7:12,])  
 SNP_freq = matrix(as.numeric(SNP_freq), nrow = nrow(SNP_freq))
 new_names <- c(paste("freq",colnames(SNP_dat)[1:3], sep = "_"),
                paste("N",colnames(SNP_dat)[1:3], sep = "_"))
 
 colnames(SNP_freq) <- new_names
 rownames(SNP_freq) <- rownames(SNP_dat)[1:6]
 write.table(SNP_freq,"SNP-frequencies.txt")
 
 # frequency of haplotype 7 at three loci:
 # interpret haplotype 7 as SNP, i.e. SNP=1 if haplotype 
 # at position is haplotype 7
 
 # positions
 haps= data.frame(idx=c(7295,7523,7850),position=dk0[c(7295,7523,7850),c(3)])
 hapsinitial.freq.k =mean(dk0[c(7295,7523,7850),16]) # haplotype 7
 initial.freq.s =mean(ds0[c(7295,7523,7850),16]) # haplotype 7
 
 haps$SNP_k_cyc0= dk0[c(7295,7523,7850),c(16)]
 haps$SNP_k_cyc6= dk6[c(7295,7523,7850),c(16)]
 haps$SNP_k_cyc12= dk12[c(7295,7523,7850),c(16)]
 haps$SNP_s_cyc0= ds0[c(7295,7523,7850),c(16)]
 haps$SNP_s_cyc6= ds6[c(7295,7523,7850),c(16)]
 haps$SNP_s_cyc12= ds12[c(7295,7523,7850),c(16)]

 write.table(haps,"three_loc_haplotype.txt")
 head(ds0,2)
 # initial frequency:
 c(initial.freq.k,initial.freq.s)
 
  