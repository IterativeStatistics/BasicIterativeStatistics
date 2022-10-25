# A propos des indices de Sobol itératifs

Comme expliqué dans [[2]](#2), le calcul des indices de Sobol de manière non-itérative s'effectue de la manière suivante :

1. Définir les lois des $p$ variables aléatoires de l'étude
2. Simuler deux échantillons de taille $n$ contenant les $p$ paramètres. On obtient ainsi deux matrices A et B de taille $n \times p$ où chaque ligne correspond à une simulation. 
3. Pour chaque $k \in \{1, \ldots, p\}$, on calcule la matrice $C^k$ égale à la matrice $A$ en remplaçant la colonne $k$ par la k-ième colonne de la matrice $B$.

Notations : on note $M_i$ la $i$-ème ligne de $M$ et $M_{[:i]}$ la matrice de taille $i \times p$ construite à partir des $i$ premières lignes de M.

Soit $Y^A_i \in \mathbb{R}$ le résultat de $f(A_i)$, où f est le modèle (~ une boîte noire) et $Y^A \in \mathbb{R}$ le résultat de $f(A)$. Même notation pour $B$ et $C^k$.

L'estimateur de Martinez permet de calculer les indices de premiers ordres de Sobol par la formule suivante :

$$
    S_k(f,A,B) = \frac{Cov(Y^B,Y^{C^k})}{\sqrt{Var(Y^B)Var(Y^{C^k})}}
$$

La méthode utilisée pour évaluer la covariance est détaillée dans [Formule covariance](docs/formula_covariance.md).

## Tester les indices de Sobol itératifs

Il existe deux fonctions utiles pour tester les indices de Sobol itératifs. 

### La fonction Ishigami

Quelques références : [ishigami](http://www.sfu.ca/~ssurjano/ishigami.html)


Dans [[1]](#1) (section 4.3), les auteurs proposent une méthode pour tester l'implémentation des indices de Sobol à l'aide de la fonction d'Ishigami.

Soit trois variables aléatoires iid $X_1$, $X_2$ et $X_3$ suivant toutes une loi uniforme sur $[- \pi, + \pi]$. La fonction d'Ishigami prise pour $a=1$, $b=7$ et $c=0.1$ donne 

$$ Y = f_{Ishig}(X_1, X_2, X_3) = \sin(X_1) + 7 \sin(X_2) + 0.1X^4_3 \sin(X_1)$$

Les valeurs théoriques des indices de Sobol à l'ordre 1 sont connues et égales à : 
$$ S_1 = 0.3139, S_2 = 0.4424 \text{ et } S_3 = 0$$

Nous utilisons ces valeurs pour tester la méthode itérative en nous appuyant sur les indices de Martinez.

## References 
<a id="1">[1]</a>  Marrel, A., Iooss, B., Laurent, B., & Roustant, O. (2009). Calculations of sobol indices for the gaussian process metamodel. Reliability Engineering & System Safety, 94(3), 742-751.

<a id="2">[2]</a>  Théophile Terraz, Alejandro Ribes, Yvan Fournier, Bertrand Iooss, and Bruno Raffin. 2017. Melissa: large scale in transit sensitivity analysis avoiding intermediate files. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '17). Association for Computing Machinery, New York, NY, USA, Article 61, 1–14. https://doi.org/10.1145/3126908.3126922