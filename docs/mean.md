# Iterative mean computation

Soit $x_n:=(x^{(i)}_n)_{i=1, \ldots, n}$ un échantillon de taille $n$. Cette échantillon est la réalisation de la variable aléatoire $X$.

On peut approximer la moyenne $\mathbb{E}\left[ X \right]$ de $X$ par son estimateur calculé sur un échantillon de taille $n$, la moyenne empirique, que l'on note $\bar{x}_n$.
$$
    \bar{x}_{n + 1} = \frac{1}{n + 1}\sum_{i=1}^{n+1} x^{(i)}_{n+1} = \frac{x^{(n+1)}_{n+1}}{n + 1} + \frac{n}{n + 1}\bar{x}_{n} = \bar{x}_{n} + \frac{x^{(n+1)}_{n+1} - \bar{x}_{n}}{n+1}
$$

On obtient ainsi une formule de récurrence pour la moyenne (qui est implémentée dans [iterative_mean.py](src/iterative_mean.py)).