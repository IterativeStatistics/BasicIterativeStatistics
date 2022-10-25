# BasicIterativeStatistics
In this repository, basic iterative statistics are implemented.


## Iterative statistics

In this repository, we implement the following basic statitistics:
- Mean (see example [here](tests/test_IterativeMean.py))
- Variance (see example [here](tests/test_IterativeVariance.py))
- Extrema (see example [here](tests/test_IterativeExtrema.py))
- Covariance (see example [here](tests/test_IterativeCovariance.py))

We detail in the [docs](docs/) folder the computation for each statistic:
- [mean](docs/mean.md)
- [covariance](docs/covariance.md)
- [sobol](docs/sobol.md)

## Additional info
L'implémentation des formules itératives s'appuient sur les papiers suivant :
- Dans [[3]](#3), l'auteur propose une méthode permettant d'évaluer la covariance (§3) de manière itérative. Un rappel est également fait sur la manière de calculer la moyenne (§1) et tous les moments d'ordre supérieurs dont la variance (§2)
- Dans [[2]](#2), les auteurs effectuent une revue des différents estimateurs permettant de calculer les indices de Sobol
- [[1]](#1) est le papier MELISSA. En ce qui concerne les statistiques itératives, les méthodes employées sont rappelées en §3 





## References 
<a id="1">[1]</a>  Théophile Terraz, Alejandro Ribes, Yvan Fournier, Bertrand Iooss, and Bruno Raffin. 2017. Melissa: large scale in transit sensitivity analysis avoiding intermediate files. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '17). Association for Computing Machinery, New York, NY, USA, Article 61, 1–14. https://doi.org/10.1145/3126908.3126922

<a id="2">[2]</a> M. Baudin, K. Boumhaout, T. Delage, B. Iooss, and J-M. Martinez. 2016. Numerical stability of Sobol' indices estimation formula. In Proceedings of the 8th International Conference on Sensitivity Analysis of Model Output (SAMO 2016). Le Tampon, Réunion Island, France.

<a id="3">[3]</a> Philippe Pébay. 2008. Formulas for robust, one-pass parallel computation of covariances and arbitrary-order statistical moments. Sandia Report SAND2008-6212, Sandia National Laboratories 94 (2008).


