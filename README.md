# DNA Modifications

N4-methylcytosine (4mC) is a kind of DNA modification which could regulate multiple biological processes. Correctly identifying 4mC sites in genomic sequences can provide precise knowledge about their genetic roles. This study aimed to develop an ensemble model to predict 4mC sites in the mouse genome. In the proposed model, DNA sequences were encoded by k-mer, enhanced nucleic acid composition and composition of k-spaced nucleic acid pairs. Subsequently, these features were optimized by using minimum redundancy maximum relevance (mRMR) with incremental feature selection (IFS) and five-fold cross-validation. The obtained optimal features were inputted into random forest classifier for discriminating 4mC from non-4mC sites in mouse. Our model could yield the overall accuracy of 79.91% on five-fold cross-validation, which was higher than the two existing models, i4mC-Mouse and 4mCpred-EL. 
# Required Package
## model_4mc requires:

Python3 (tested 3.5.4)

numpy (tested 1.18.1)

pandas (tested 1.0.1)

jupyter (tested 1.0.0)

scikit-learn (tested 0.22.1)
