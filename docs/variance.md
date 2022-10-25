# Iterative variance

Soit $x_n:=(x^{(i)}_n)_{i=1, \ldots, n}$  un échantillon de taille $n$. Cette échantillon est la réalisation de la variable aléatoire $X$.

On peut approximer la variance $Var(X)$ par la formule de variance empirique $s^2$:
$$ s^2_n = \frac{1}{n-1}  \sum_{i = 1}^{n} (x_i - \bar{x_n})^2$$
        
where \bar{x} is the sample mean.

Cette méthode est implémentée dans openturns.