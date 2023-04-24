# Iterative quantile computation

Cette bibliothèque implémente un Algorithme de Robbins-Monro (RM) pour l'estimation des quantiles. Les paramètres de réglage de cet algorithme ont été étudiés (par le biais de tests numériques intensifs) dans l'article suivant:

Iooss, Bertrand, and Jérôme Lonchampt. "Robust tuning of Robbins-Monro algorithm for quantile estimation-Application to wind-farm asset management." ESREL 2021. 2021.

Dans l'algorithme implémenté, le nombre final d'itérations (c'est-à-dire le nombre d'exécutions du modèle informatique) N est a priori fixe, ce qui est une façon classique de traiter avec des problèmes de quantification de l'incertitude.
