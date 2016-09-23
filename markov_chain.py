#function
def markov_chain(n, pa, pb, a0=1):      #pa=p(A), pb=p(B)
    if(n==0): return a0
    if(n==1): return pa
    else: return ((pa*(markov_chain(n-1, pa, pb)))+((1-pb)*(1-(markov_chain(n-1, pa, pb)))))
                  #P(An|A1)*P(An-1)+((1-P(An|B1))*(1-P(An-1))) Theorem of Total Probability


#examples
print markov_chain(3, 0.6, 0.8)
print markov_chain(3, 0.5, 0)
