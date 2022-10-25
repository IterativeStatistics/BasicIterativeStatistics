# Iterative covariance
Soit $x_n:=(x^{(i)}_n)_{i=1, \ldots, n}$  et $y_n:=(y^{(i)}_n)_{i=1, \ldots, n}$ deux échantillons de taille $n$. Ces échantillon sont la réalisation de deux variables aléatoires $X$ et $Y$.

On peut approximer la covariance $cov(X,Y)$ par la formule de covariance empirique $\bar{cov}(x,y)$:

$$
    cov(x_{n+1}, y_{n+1}) = \frac{1}{n}\sum_{i = 1}^{n+1}(x^{(i)}_{n+1} - \bar{x}_{n + 1})(y^{(i)}_{n+1} - \bar{y}_{n+1})
    = \frac{1}{n}\sum_{i = 1}^{n+1}(x^{(i)}_{n+1} - \bar{x}_{n})(y^{(i)}_{n+1} - \bar{y}_{n+1}) + \frac{1}{n}\sum_{i = 1}^{n +1}(\bar{x}_{n} - \bar{x}_{n+1})(y^{(i)}_{n+1} - \bar{y}_{n+1}) 
$$

Or, 
$$
\sum_{i = 1}^{n + 1}(\bar{x}_{n} - \bar{x}_{n+1})(y^{(i)}_{n+1} - \bar{y}_{n+1}) = 
(\bar{x}_{n} - \bar{x}_{n+1})\sum_{i = 1}^{n + 1}(y^{(i)}_{n+1} - \bar{y}_{n+1}) = (\bar{x}_{n} - \bar{x}_{n+1}) [ (n+1)\bar{y}_{n+1} - (n+1)\bar{y}_{n+1} ] = 0
$$

D'où, 
$$
    cov(x_{n+1}, y_{n+1}) 
    = \frac{1}{n}\sum_{i = 1}^{n+1}(x^{(i)}_{n+1} - \bar{x}_{n})(y^{(i)}_{n+1} - \bar{y}_{n+1}) 
    = \frac{1}{n}\sum_{i = 1}^{n+1}(x^{(i)}_{n+1} - \bar{x}_{n})(y^{(i)}_{n+1} - \bar{y}_{n}) + \frac{1}{n}\sum_{i = 1}^{n+1}(x^{(i)}_{n+1} - \bar{x}_{n})(\bar{y}_{n} - \bar{y}_{n+1})
$$

Or, 
$$
 \sum_{i = 1}^{n+1}(x^{(i)}_{n+1} - \bar{x}_{n})(\bar{y}_{n} - \bar{y}_{n+1}) =  (\bar{y}_{n} - \bar{y}_{n+1})\sum_{i = 1}^{n+1}(x^{(i)}_{n+1} - \bar{x}_{n}) = (\bar{y}_{n} - \bar{y}_{n+1}) [(n+1)\bar{x}_{n + 1} - (n+1)\bar{x}_{n} ] \\
 = (n+1)(\bar{y}_{n} - \bar{y}_{n+1}) (\bar{x}_{n + 1} - \bar{x}_{n} ) 
$$

On déduit

$$
    cov(x_{n+1}, y_{n+1}) 
    = \frac{1}{n}\sum_{i = 1}^{n+1}(x^{(i)}_{n+1} - \bar{x}_{n})(y^{(i)}_{n+1} - \bar{y}_{n}) + \frac{n+1}{n}(\bar{y}_{n} - \bar{y}_{n+1}) (\bar{x}_{n + 1} - \bar{x}_{n} ) \\
    = \frac{1}{n}\sum_{i = 1}^{n}(x^{(i)}_{n+1} - \bar{x}_{n})(y^{(i)}_{n+1} - \bar{y}_{n}) + \frac{1}{n}(x^{(n+1)}_{n+1} - \bar{x}_{n})(y^{(n+1)}_{n+1} - \bar{y}_{n}) + \frac{n+1}{n}(\bar{y}_{n} - \bar{y}_{n+1}) (\bar{x}_{n + 1} - \bar{x}_{n} ) \\
    = \frac{n-1}{n}cov(x_{n}, y_{n})  + \frac{1}{n}(x^{(n+1)}_{n+1} - \bar{x}_{n})(y^{(n+1)}_{n+1} - \bar{y}_{n}) - \frac{n+1}{n}(\bar{y}_{n} - \bar{y}_{n+1}) (  \bar{x}_{n} - \bar{x}_{n + 1} )
$$

Or, 
$$
    \bar{x}_{n} - \bar{x}_{n+1} = \frac{1}{n} \sum_{i = 1}^{n}x^{(i)}_{n} - \frac{1}{n + 1} \sum_{i = 1}^{n + 1}x^{(i)}_{n+1} = \frac{1}{ n(n + 1)} \sum_{i = 1}^{n}x^{(i)}_{n}- \frac{1}{n + 1}x^{(n+1)}_{n+1} = \frac{1}{ n + 1} [\bar{x}_n - {(n+1)}x_{n+1}^{(n+1)}]
$$
On déduit alors 
$$
    cov(x_{n+1}, y_{n+1}) 
    = \frac{n-1}{n}cov(x_{n}, y_{n})  + \frac{1}{n}(x^{(n+1)}_{n+1} - \bar{x}_{n})(y^{(n+1)}_{n+1} - \bar{y}_{n}) + \frac{1}{n(n+1)}(\bar{y}_n - {(n+1)}y_{n+1}^{(n+1)}) (\bar{x}_n - {(n+1)}x_{n+1}^{(n+1)})
$$

Or, 
$$
  (x^{(n+1)}_{n+1} - \bar{x}_{n})(y^{(n+1)}_{n+1} - \bar{y}_{n}) + \frac{1}{n+1}({(n+1)}x_{n+1}^{(n+1)} - \bar{x}_n)  ({(n+1)}y_{n+1}^{(n+1)} - \bar{y}_n)  
$$

## Calcul 2

$$
    cov(x_{n+1}, y_{n+1}) 
    = \frac{1}{n}\sum_{i = 1}^{n+1}(x^{(i)}_{n+1} - \bar{x}_{n+1})(y^{(i)}_{n+1} - \bar{y}_{n+1})  
$$

## Vérification des formules : cas où $X = Y$

$$
    cov(x_{n+1}, y_{n+1}) = cov(x_{n+1}, x_{n+1}) = var(x_{n+1})
    = \frac{n-1}{n}var(x_{n}, y_{n})  + \frac{1}{n}(x^{(n+1)}_{n+1} - \bar{x}_{n})^2 + \frac{n+1}{n}(\bar{x}_{n + 1} - \bar{x}_{n} )^2
$$

Attendu

$$
    var(x_{n+1}) = \frac{1}{n-1}var(x_{n}) + \frac{n + 1}{n^2} (x^{(n+1)}_{n+1} - \bar{x}_{n + 1})^2
$$

**Preuve**

$$
  var(x_{n+1}) = \frac{1}{n}\sum_{i = 1}^{n+1} (x^{(i)}_{n+1} - \bar{x}_{n + 1})^2  
$$

Using that
$$
    (x^{(i)}_{n+1} - \bar{x}_{n + 1})^2 = \left[(x^{(i)}_{n+1} - \bar{x}_{n}) + (\bar{x}_{n} - \bar{x}_{n + 1})\right]^2 = (x^{(i)}_{n+1} - \bar{x}_{n})^2 + 2(x^{(i)}_{n+1} - \bar{x}_{n})(\bar{x}_{n} - \bar{x}_{n + 1}) + (\bar{x}_{n} - \bar{x}_{n + 1})^2
$$
We deduce that
$$
    var(x_{n+1}) = \frac{1}{n}\sum_{i = 1}^{n+1}(x^{(i)}_{n+1} - \bar{x}_{n})^2 + \frac{2(\bar{x}_{n} - \bar{x}_{n + 1})}{n}\sum_{i = 1}^{n+1}(x^{(i)}_{n+1} - \bar{x}_{n}) + \frac{1}{n}\sum_{i = 1}^{n+1}(\bar{x}_{n} - \bar{x}_{n + 1})^2 \\
    = \frac{n-1}{n} var(x_{n}) + \frac{1}{n}(x^{(n+1)}_{n+1} - \bar{x}_{n})^2 + \frac{(\bar{x}_{n} - \bar{x}_{n + 1})}{n}\sum_{i = 1}^{n+1}\left[2x^{(i)}_{n+1} - 2\bar{x}_{n} + \bar{x}_{n} - \bar{x}_{n + 1} \right]
$$

Or, 
$$
    \sum_{i = 1}^{n+1}\left[2x^{(i)}_{n+1} - 2\bar{x}_{n} + \bar{x}_{n} - \bar{x}_{n + 1} \right] = \sum_{i = 1}^{n+1}\left[x^{(i)}_{n+1} - \bar{x}_{n} \right] + \sum_{i = 1}^{n+1}\left[x^{(i)}_{n+1} - \bar{x}_{n+1} \right] = \sum_{i = 1}^{n+1}\left[x^{(i)}_{n+1} - \bar{x}_{n} \right]  \\
    = \sum_{i = 1}^{n}\left[x^{(i)}_{n+1} - \bar{x}_{n} \right]  + \left[x^{(n+1)}_{n+1} - \bar{x}_{n} \right]  = \left[x^{(n+1)}_{n+1} - \bar{x}_{n} \right]

$$

D'où, 
$$
    var(x_{n+1}) = \frac{n-1}{n} var(x_{n}) + \frac{1}{n}(x^{(n+1)}_{n+1} - \bar{x}_{n})\left[ x^{(n+1)}_{n+1} - \bar{x}_{n} + \bar{x}_{n} - \bar{x}_{n + 1} \right] \\
    = \frac{n-1}{n} var(x_{n}) + \frac{1}{n}(x^{(n+1)}_{n+1} - \bar{x}_{n})\left[ x^{(n+1)}_{n+1} - \bar{x}_{n + 1} \right]
$$

