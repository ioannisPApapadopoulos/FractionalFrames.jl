# FractionalFrames.jl

This repository implements the numerical examples found in two manuscripts:

(1) "Explicit fractional Laplacians and Riesz potentials of classical functions", Timon S. Gutleb, Ioannis. P. A. Papadopoulos (2023).

(2) "A frame approach for equations involving the fractional Laplacian", Ioannis. P. A. Papadopoulos, Timon S. Gutleb, José A. Carrillo, and Sheehan Olver (2023).

We numerically approximate the solution to equations involving fractional Laplacian operators via frame approach. This approach reduces solving an equation with nonlocal terms to an interpolation problem for the right-hand side. We find expansions via a truncated SVD which alleviates the perceived ill-conditioning.

### Explicit fractional Laplacians and Riesz potentials of classical functions
|Figure|File: examples/|
|:-:|:-:|
|2|[weighted-zernike.jl](https://github.com/ioannisPApapadopoulos/FractionalFrames.jl/blob/main/examples/weighted-zernike.jl)|

### A frame approach for equations involving the fractional Laplacian
|Figure|File: examples/|
|:-:|:-:|
|1|[gaussain.jl](https://github.com/ioannisPApapadopoulos/FractionalFrames.jl/blob/main/examples/gaussian.jl)|
|2|[multiple-exponents.jl](https://github.com/ioannisPApapadopoulos/FractionalFrames.jl/blob/main/examples/multiple-exponents.jl)|
|3|[2d-gaussian.jl](https://github.com/ioannisPApapadopoulos/FractionalFrames.jl/blob/main/examples/2d-gaussian.jl)|
|4a|[fractional-heat-fundamental-runge-kutta.jl](https://github.com/ioannisPApapadopoulos/FractionalFrames.jl/blob/main/examples/fractional-heat-fundamental-runge-kutta.jl)|
|4b|[tails.jl](https://github.com/ioannisPApapadopoulos/FractionalFrames.jl/blob/main/examples/tails.jl)|
|5|[variable-s-runge-kutta.jl](https://github.com/ioannisPApapadopoulos/FractionalFrames.jl/blob/main/examples/variable-s-runge-kutta.jl)|

## Contact
Ioannis Papadopoulos: papadopoulos@wias-berlin.de