---
header-includes:
    - \usepackage[vmargin=0.5in,hmargin=0.5in]{geometry}
    - \usepackage{breqn}
---


# Question 1


# Question 2


# Question 3a

### Step 1:

At step 1 we observe a reading for $Z$ of 2. Let $t = 1$. We solve for the MAP as follows:

\begin{equation} \bar{bel}(X_t = 1) = P(X_t = 1 | X_o = 1) + P(X_t = 1 | X_o = 2) + P(X_t = 1 | X_o = 3) \end{equation}
\begin{equation} = P(X_t = 1 | X_o = 1) bel(X_o = 1) + P(X_t = 1 | X_o = 2)  bel(X_o = 2) + P(X_t = 1 | X_o = 3)  bel(X_o = 3) \end{equation}
\begin{equation} = (0.2)(1/3)+(0.2)(1/3)+(0.4)(1/3) = 0.26 \end{equation}
\begin{equation} bel(X_t = 1) = Ŋ P(Z_t = 2 | X_t = 1) \bar{bel}(X_t = 1) = Ŋ \times 0.1 \times 0.26 = \NG 0.026 \end{equation}


\begin{equation} \bar{bel}(X_t = 2) = P(X_t = 2 | X_o = 1) + P(X_t = 2 | X_o = 2) + P(X_t = 2 | X_o = 3)\end{equation}
\begin{equation} = P(X_t = 2 | X_o = 1) bel(X_o = 1) + P(X_t = 2 | X_o = 2)  bel(X_o = 2) + P(X_t = 2 | X_o = 3)  bel(X_o = 3)\end{equation}
\begin{equation} = (0.3)(1/3)+(0.7)(1/3)+(0.2)(1/3) = 0.3\dot{9}\end{equation}
\begin{equation} bel(X_t = 2) = Ŋ P(Z_t = 2 | X_t = 2) \bar{bel}(X_t = 2) = Ŋ \times 0.7 \times 0.39 = \NG 0.272\dot{9} \end{equation}


\begin{equation} \bar{bel}(X_t = 3) = P(X_t = 4 | X_o = 1) + P(X_t = 4 | X_o = 2) + P(X_t = 4 | X_o = 3)\end{equation}
\begin{equation} = P(X_t = 3| X_o = 1) bel(X_o = 1) + P(X_t = 3 | X_o = 2)  bel(X_o = 2) + P(X_t = 3 | X_o = 3)  bel(X_o = 3)\end{equation}
\begin{equation} = (0.5)(1/3)+(0.1)(1/3)+(0.4)(1/3) = 0.\dot{3}\end{equation}
\begin{equation} bel(X_t = 3) = Ŋ P(Z_t = 2 | X_t = 3) \bar{bel}(X_t = 3) = Ŋ \times 0.3 \times 0.3 = \NG 0.0\dot{9} \end{equation}

Therefore:

\begin{equation} \NG = (0.0\dot{9} + 0.026 + 0.272\dot{9} ) ^{-1} = 0.398 \end{equation}

This implies:

\begin{equation} bel(X_1 = 1) \approx 0.065, bel(X_1 = 2) \approx 0.684, bel(X_1 = 3) \approx 0.25 \end{equation}

The MAP is therefore $X_1 = 2$.

### Step 2:

At step 2 we observe a reading for $Z$ of 3. Let $t = 2$. We solve for the MAP as follows:

\begin{equation} \bar{bel}(X_t = 1) = P(X_t = 1 | X_1 = 1) + P(X_t = 1 | X_1 = 2) + P(X_t = 1 | X_1 = 3)\end{equation}
\begin{equation} = P(X_t = 1 | X_1 = 1) bel(X_1 = 1) + P(X_t = 1 | X_1 = 2)  bel(X_1 = 2) + P(X_t = 1 | X_1 = 3)  bel(X_1 = 3)\end{equation}

\begin{equation} = (0.2)(0.065)+(0.2)(0.684)+(0.4)(0.25) = 0.2498 \end{equation}
\begin{equation} bel(X_t = 1) = Ŋ P(Z_t = 2 | X_t = 1) \bar{bel}(X_t = 1) = Ŋ \times 0.1 \times 0.2498 = \NG 0.02498 \end{equation}

\begin{equation} \bar{bel}(X_t = 2) = P(X_t = 2 | X_1 = 1) + P(X_t = 2 | X_1 = 2) + P(X_t = 2| X_1 = 3)\end{equation}
\begin{equation} = P(X_t = 2 | X_1 = 1) bel(X_1 = 1) + P(X_t = 2 | X_1 = 2)  bel(X_1 = 2) + P(X_t = 2 | X_1 = 3)  bel(X_1 = 3)\end{equation}
\begin{equation} = (0.3)(0.065)+(0.7)(0.684) + (0.2)(0.25)  = 0.5483 \end{equation}

\begin{equation} bel(X_t = 2) = Ŋ P(Z_t = 2 | X_t = 2) \bar{bel}(X_t = 2) = Ŋ \times 0.7 \times 0.5483 = \NG 0.38381\end{equation}


\begin{equation} \bar{bel}(X_t = 3) = P(X_t = 3 | X_1 = 1) + P(X_t = 3 | X_1 = 2) + P(X_t = 3| X_1 = 3)\end{equation}
\begin{equation} = P(X_t = 3 | X_1 = 1) bel(X_1 = 1) + P(X_t = 3 | X_1 = 2)  bel(X_1 = 2) + P(X_t = 3 | X_1 = 3)  bel(X_1 = 3)\end{equation}
\begin{equation} = (0.5)(0.065)+(0.1)( 0.684)+(0.4)(0.25)  = 0.2 \end{equation}

\begin{equation} bel(X_t = 3) = Ŋ P(Z_t = 3 | X_t = 2) \bar{bel}(X_t = 3) = Ŋ \times 0.2 \times 0.2 = \NG 0.04  \end{equation}

The normalization constant is thus: $ (0.02498 + 0.38381 + 0.04)^{-1} = 2.22821 $ So the beliefs are: \begin{equation} bel(X_2 = 1) \approx 0.055, bel(X_2 = 2) \approx 0.85, bel(X_2 = 3) \approx 0.089 \end{equation}

The MAP is therefore $X_2 = 2$.

### Step 3:

At step 3 we observe a reading for $Z$ of 1. Let $t = 3$. We solve for the MAP as follows:


\begin{equation} \bar{bel}(X_t = 1) = P(X_t = 1 | X_2 = 1) + P(X_t = 1 | X_2 = 2) + P(X_t = 1 | X_2 = 3)\end{equation}
\begin{equation} = P(X_t = 1 | X_2 = 1) bel(X_2 = 1) + P(X_t = 1 | X_2 = 2)  bel(X_2 = 2) + P(X_t = 1 | X_2 = 3)  bel(X_2 = 3)\end{equation}

\begin{equation} = (0.2)(0.055)+(0.2)(0.85)+(0.4)(0.089) = 0.2166 \end{equation}
\begin{equation} bel(X_t = 1) = Ŋ P(Z_t = 1 | X_t = 1) \bar{bel}(X_t = 1) = Ŋ \times 0.6 \times 0.2166 = \NG 0.12996 \end{equation}


\begin{equation} \bar{bel}(X_t = 2) = P(X_t = 2 | X_2 = 1) + P(X_t = 2 | X_2 = 2) + P(X_t = 2 | X_2 = 3)\end{equation}
\begin{equation} = P(X_t = 2 | X_2 = 1) bel(X_2 = 1) + P(X_t = 2 | X_2 = 2)  bel(X_2 = 2) + P(X_t = 2 | X_2 = 3)  bel(X_2 = 3)\end{equation}

\begin{equation} = (0.2)(0.055)+(0.7)(0.85)+(0.2)(0.089) = 0.6238 \end{equation}
\begin{equation} bel(X_t = 2) = Ŋ P(Z_t = 1 | X_t = 2) \bar{bel}(X_t = 2) = Ŋ \times 0.1 \times 0.6238 = \NG 0.06238 \end{equation}

\begin{equation} \bar{bel}(X_t = 3) = P(X_t = 3 | X_2 = 1) + P(X_t = 3 | X_2 = 2) + P(X_t = 3 | X_2 = 3)\end{equation}
\begin{equation} = P(X_t = 3 | X_2 = 1) bel(X_2 = 1) + P(X_t = 3 | X_2 = 2)  bel(X_2 = 2) + P(X_t = 3 | X_2 = 3)  bel(X_2 = 3)\end{equation}

\begin{equation} = (0.5)(0.055)+(0.1)(0.85)+(0.4)(0.089) = 0.1481 \end{equation}
\begin{equation} bel(X_t = 2) = Ŋ P(Z_t = 1 | X_t = 3) \bar{bel}(X_t = 2) = Ŋ \times 0.2 \times 0.1481 = \NG 0.02962 \end{equation}

The normalization constant is \begin{equation} (0.12996 + 0.06238 + 0.02962)^{-1} = 4.50532 \end{equation}. So the beliefs are: \begin{equation} bel(X_3 = 1) \approx 0.5855, bel(X_3 = 2) \approx 0.281, bel(X_3 = 3) \approx 0.133 \end{equation}

The MAP is therefore $X_3 = 1$.


Therefore, \begin{equation} arg_{X_t} max P(X_t | Z^t) , t \in \{1,2,3\} = \{2,2,1\}\end{equation}


# Question 3b

For the forward step:^[I round to two significant digits.]

\begin{equation} \alpha_0 (1)  = P(Z_0 = 2 | X_0 = 1) P(X_0 = 1) = \NG_0 (0.1)(1/3)  = 0.09 \end{equation}
\begin{equation} \alpha_0 (2)  = P(Z_0 = 2 | X_0 = 2) P(X_0 = 2) = \NG_0 (0.7)(1/3) = 0.64 \end{equation}
\begin{equation} \alpha_0 (3)  = P(Z_0 = 2 | X_0 = 3) P(X_0 = 3) = \NG (0.3)(1/3)  = 0.27 \end{equation}

Where \begin{equation} \NG_0 = (\alpha_0 (1) + \alpha_0 (2) + \alpha_0 (3))^{-1}  = 2.72727 \end{equation}

\begin{equation} \alpha_1 (1) = P(Z_1 = 3 | X_1 = 1) [ P(X_1 = 1 | X_0 = 1) \alpha_0 (1) + P(X_1 = 1 | X_0 = 2) \alpha_0 (2) + P(X_1 = 1 | X_0 = 3) \alpha_0 (3) ] \end{equation}
\begin{equation} = (0.6) [ (0.2)(0.09) + (0.2)(0.64)  + (0.4)(0.27)] = \NG_1 0.15  = 0.41 \end{equation}

\begin{equation} \alpha_1 (2) = P(Z_1 = 3 | X_1 = 2) [ P(X_1 = 2 | X_0 = 1) \alpha_0 (1) + P(X_1 = 2 | X_0 = 2) \alpha_0 (2) + P(X_1 = 2 | X_0 = 3) \alpha_0 (3) ] \end{equation}
\begin{equation} = (0.2) [ (0.3)(0.09) + (0.7)(0.64)  + (0.2)(0.27)] = \NG_1 0.105 = 0.29 \end{equation}

\begin{equation} \alpha_1 (3) = P(Z_1 = 3 | X_1 = 3) [ P(X_1 = 3 | X_0 = 1) \alpha_0 (1) + P(X_1 = 3 | X_0 = 2) \alpha_0 (2) + P(X_1 = 3 | X_0 = 3) \alpha_0 (3) ] \end{equation}
\begin{equation} = (0.5) [ (0.5)(0.09) + (0.1)(0.64)  + (0.4)(0.27)] = \NG_1 0.108  = 0.30 \end{equation}

Where \begin{equation} \NG_1 = (\alpha_1 (1) + \alpha_1 (2) + \alpha_1 (3))^{-1}  = 2.75482 \end{equation}

\begin{equation} \alpha_2 (1) = P(Z_2 = 1 | X_2 = 1) [ P(X_2 = 1 | X_1 = 1) \alpha_1 (1) + P(X_2 = 1 | X_1 = 2) \alpha_1 (2) + P(X_2 = 1 | X_1 = 3) \alpha_1 (3) ] \end{equation}
\begin{equation} = (0.6) [ (0.2)(0.41) + (0.2)(0.29)   + (0.4)(0.3)] = \NG_2 0.156 = 0.63 \end{equation}

\begin{equation} \alpha_2 (2) = P(Z_2 = 1 | X_2 = 2) [ P(X_2 = 2 | X_1 = 1) \alpha_1 (1) + P(X_2 = 2 | X_1 = 2) \alpha_1 (2) + P(X_2 = 2 | X_1 = 3) \alpha_1 (3) ] \end{equation}
\begin{equation} = (0.1) [ (0.3)(0.41) + (0.7)(0.29)   + (0.2)(0.3)] = \NG_2 0.0386 =  0.16 \end{equation}

\begin{equation} \alpha_2 (3) = P(Z_2 = 1 | X_2 = 3) [ P(X_2 = 3 | X_1 = 1) \alpha_1 (1) + P(X_2 = 3 | X_1 = 2) \alpha_1 (2) + P(X_2 = 3 | X_1 = 3) \alpha_1 (3) ] \end{equation}
\begin{equation} = (0.2) [ (0.3)(0.41) + (0.1)(0.29)   + (0.4)(0.3)] = \NG_2 0.0544 =  0.22 \end{equation}

Where we calculate the normalization constant as above, \begin{equation} \NG_2 = 4.01606 \end{equation}

For the backward step, define:

\begin{equation} \beta_2 (1) = 1, \beta_2 (2) = 1, \beta_2 (3) = 1 \end{equation}

This can be normalized to sum to 1 by multiplying it by a normalization constant \begin{equation} \NG_2 = 1/3 \end{equation}

With this as our starting point, we calculate:

\begin{equation}
\begin{aligned}
\beta_1 (1) &= P( Z_2 = 1 | X_2 = 1 ) P( X_2 = 1 | X_1 = 1) \beta_2(1) +  P( Z_2 = 1 | X_2 = 2 ) P( X_2 = 2 | X_1 = 1) \beta_2(2)
\\& + P( Z_2 = 1 | X_2 = 3 ) P( X_2 = 3 | X_1 = 1) \beta_2(3)  
\end{aligned}
\end{equation}

\begin{equation} = (0.6)(0.2)(1/3) + (0.1)(0.2)(1/3) + (0.2)(0.5)(1/3) = \NG_1 0.08  = 0.31 \end{equation}

\begin{equation}
\begin{aligned} \beta_1 (2) &= P( Z_2 = 1 | X_2 = 1 ) P( X_2 = 1 | X_1 = 2) \beta_2(1) +  P( Z_2 = 1 | X_2 = 2 ) P( X_2 = 2 | X_1 = 2) \beta_2(2)
\\&+  P( Z_2 = 1 | X_2 = 3 ) P( X_2 = 3 | X_1 = 2) \beta_2(3)
\\&
\\&= (0.6)(0.2)(1/3) + (0.1)(0.7)(1/3) + (0.2)(0.1)(1/3) = \NG_1 0.07  = 0.27
\end{aligned}
\end{equation}

\begin{equation}
\begin{aligned}\beta_1 (3) &= P( Z_2 = 1 | X_2 = 1 ) P( X_2 = 1 | X_1 = 3) \beta_2(1) +  P( Z_2 = 1 | X_2 = 2 ) P( X_2 = 2 | X_1 = 3) \beta_2(2)
\\&+  P( Z_2 = 1 | X_2 = 3 ) P( X_2 = 3 | X_1 = 3) \beta_2(3)  
\\&
\\&= (0.6)(0.4)(1/3) + (0.1)(0.2)(1/3) + (0.2)(0.4)(1/3) = \NG_1 0.11  = 0.42
\end{aligned}
 \end{equation}

Where  \begin{equation}  \NG_2 = 3.84615 \end{equation}

\begin{equation}
\begin{aligned} \beta_0 (1) &= P( Z_1 = 1 | X_1 = 1 ) P( X_1 = 1 | X_0 = 1) \beta_1(1) +  P( Z_1 = 1 | X_1 = 2 ) P( X_1 = 2 | X_0 = 1) \beta_1(2)
\\&+  P( Z_1 = 1 | X_1 = 3 ) P( X_1 = 3 | X_0 = 1) \beta_1(3)
\\
\\&= (0.3)(0.2)(0.31) + (0.2)(0.3)(0.27) + (0.5)(0.5)(0.42) = \NG_2 0.1398  = 0.40
\end{aligned}
\end{equation}

\begin{equation}
\begin{aligned} \beta_0 (2) &= P( Z_1 = 1 | X_1 = 1 ) P( X_1 = 1 | X_0 = 2) \beta_1(1) +  P( Z_1 = 1 | X_1 = 2 ) P( X_1 = 2 | X_0 = 2) \beta_1(2)
\\&+  P( Z_1 = 1 | X_1 = 3 ) P( X_1 = 3 | X_0 = 2) \beta_1(3)  
\\
\\&= (0.3)(0.2)(0.31) + (0.2)(0.7)(0.27) + (0.5)(0.1)(0.42) = \NG_2 0.0774 = 0.22
\end{aligned}
 \end{equation}

\begin{equation}
\begin{aligned} \beta_0 (3) &= P( Z_1 = 1 | X_1 = 1 ) P( X_1 = 1 | X_0 = 3) \beta_1(1) +  P( Z_1 = 1 | X_1 = 2 ) P( X_1 = 2 | X_0 = 3) \beta_1(2)
\\&+  P( Z_1 = 1 | X_1 = 3 ) P( X_1 = 3 | X_0 = 3) \beta_1(3)  
\\
\\&= (0.3)(0.4)(0.31) + (0.2)(0.2)(0.27) + (0.5)(0.4)(0.42) = \NG_2 0.132 = 0.38
\end{aligned}
\end{equation}

Where $$ \NG_2 = 2.86369 $$

From these we calculate  $$ P(Z^{T})$$

$$ P(Z^{T}) = \sum_{X_T} \alpha(X_T) \beta(X_T) = \sum_{X_T} \alpha(X_T) $$

because $$ \beta(X_T) = 1 $$

\begin{equation}
\sum_{X_t = 0}^2 \alpha(X_t)  = \alpha(X_0 = 2) + \alpha(X_1 = 3) + \alpha(X_2 = 1) = 0.64 + 0.3 + 0.63  = 1.57
\end{equation}

Which we use to calculate the probability of each state at a given point in time.

\begin{equation}
P(X_t | Z^T) = \frac{\alpha(X_t)\beta(X_t)}{P(Z^T)}
\end{equation}

For Step 0:

\begin{equation}
P(X_0 = 1 | Z^T) = \frac{\alpha_0(1)\beta_0(1)}{P(Z^T)} = \frac{0.09 \times 0.40}{1.57} = 0.023
\end{equation}

\begin{equation}
P(X_0 = 2 | Z^T) = \frac{\alpha_0(2)\beta_0(2)}{P(Z^T)} = \frac{0.64 \times 0.22}{1.57} = 0.089
\end{equation}

\begin{equation}
P(X_0 = 3 | Z^T) = \frac{\alpha_0(3)\beta_0(3)}{P(Z^T)} = \frac{0.27 \times 0.38}{1.57} = 0.065
\end{equation}

Once again, we normalize and this produces probabilities of:

$$ P(X_0 = 1 | Z^T) = 0.13, P(X_0 = 2 | Z^T) = 0.5, P(X_0 = 3 | Z^T)  = 0.37 $$

So the MAP estimate for $t = 0$ is state 1. Repeating this process for steps 1 and 2, we obtain the following estimates:

$$ P(X_1 = 1 | Z^T) = \NG 0.081 = 0.12, P(X_1 = 2 | Z^T) = \NG 0.5 = 0.75, P(X_1 = 3 | Z^T)  = \NG  0.08  = 0.12 $$
$$ P(X_2 = 1 | Z^T) = \NG 0.13 = 0.62 , P(X_2 = 1 | Z^T) =  \NG 0.034 = 0.16 , P(X_2 = 2 | Z^T)  = \NG 0.047  = 0.22 $$

Therefore, \begin{equation} arg_{X_t} max P(X_t | Z^T) = \{1,2,1\}\end{equation}

# Question 3c

First we solve for the $\delta_0(i), i \in \{1,2,3\}$:

$$ \delta_0(1) = P(Z_0 = 2 | X_0 = 1)P(X_0 = 1) = 0.1 \times (1/3) $$
$$ \delta_0(2) = P(Z_0 = 2 | X_0 = 2)P(X_0 = 2) = 0.7 \times (1/3) $$
$$ \delta_0(3) = P(Z_0 = 2 | X_0 = 3)P(X_0 = 3) = 0.3 \times (1/3) $$

We then evaluate $\delta_1(i), i \in \{1,2,3\}$:
\begin{equation}
\begin{aligned}
\delta_1(1) &= P(Z_1 = 3 | X_1 = 1) argmax (priors),
\\& priors = \{ P(X_1 = 1 | X_0 = 1) \delta_0(1) = (0.2)(\delta_0(1)),
\\& P(X_1 = 1 | X_0 = 2) \delta_0(2) = (0.2)(\delta_0(2)),
\\& P(X_1 = 1 | X_0 = 3) \delta_0(3) = (0.4)(\delta_0(3)) \}
\end{aligned}
\end{equation}

\begin{equation}
\begin{aligned}
\delta_1(2) &= P(Z_1 = 3 | X_1 = 2) argmax (priors),
\\& priors = \{ P(X_1 = 2 | X_0 = 1) \delta_0(1) = (0.3)(\delta_0(1)),
\\& P(X_1 = 2 | X_0 = 2) \delta_0(2) = (0.7)(\delta_0(2)),
\\& P(X_1 = 2 | X_0 = 3) \delta_0(3) = (0.2)(\delta_0(3)) \}
\end{aligned}
\end{equation}

\begin{equation}
\begin{aligned}
\delta_1(3) &= P(Z_1 = 3 | X_1 = 3) argmax (priors),
\\& priors = \{ P(X_1 = 3 | X_0 = 1) \delta_0(1) = (0.5)(\delta_0(1)),
\\& P(X_1 = 3 | X_0 = 2) \delta_0(2) = (0.1)(\delta_0(2)),
\\& P(X_1 = 3 | X_0 = 3) \delta_0(3) = (0.4)(\delta_0(3)) \}
\end{aligned}
\end{equation}

Next, we repeat this procedure for time step 2:

\begin{equation}
\begin{aligned}
\delta_2(1) &= P(Z_2 = 1 | X_1 = 1) argmax (priors),
\\& priors = \{ P(X_1 = 1 | X_0 = 1) \delta_1(1) = (0.2)(\delta_1(1)),
\\& P(X_1 = 1 | X_0 = 2) \delta_1(2) = (0.2)(\delta_1(2)),
\\& P(X_1 = 1 | X_0 = 3) \delta_1(3) = (0.4)(\delta_1(3)) \}
\end{aligned}
\end{equation}

\begin{equation}
\begin{aligned}
\delta_2(2) &= P(Z_2 = 1 | X_1 = 2) argmax (priors),
\\& priors = \{ P(X_1 = 2 | X_0 = 1) \delta_1(1) = (0.2)(\delta_1(1)),
\\& P(X_1 = 2 | X_0 = 2) \delta_1(2) = (0.7)(\delta_1(2)),
\\& P(X_1 = 2 | X_0 = 3) \delta_1(3) = (0.2)(\delta_1(3)) \}
\end{aligned}
\end{equation}

\begin{equation}
\begin{aligned}
\delta_2(3) &= P(Z_2 = 1 | X_1 = 3) argmax (priors),
\\& priors = \{ P(X_1 = 3 | X_0 = 1) \delta_1(1) = (0.5)(\delta_1(1)),
\\& P(X_1 = 3 | X_0 = 2) \delta_1(2) = (0.1)(\delta_1(2)),
\\& P(X_1 = 3 | X_0 = 3) \delta_1(3) = (0.4)(\delta_1(3)) \}
\end{aligned}
\end{equation}


The MAP sequence we obtain from this is $\{X_0 = 3, X_1 = 3,X_2 = 1\}$. This compares to the filtering sequence $\{X_0 = 2, X_1 = 3,X_2 = 1\}$ and the smoothing sequence $\{X_0 = 1, X_1 = 2,X_2 = 1\}$.

# Question 4a

I was not able to complete this question. I have submitted what I was able to produce.

# Question 4b

I was not able to complete this question.


# Question 5a

I worked briefly (1 hour - 30 min) with Joe Denby and Ben Pick.

# Question 5b

I spent roughly 29 hours on the problem set from Monday 22 - 29 April.
