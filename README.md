# SHM for Under Damped Regions PINNs

This repository contains two implementations of Physics-Informed Neural Networks (PINNs):

1. Burgers equation with standard and Fourier feature PINNs  
2. Underdamped harmonic oscillator as a parametric PINN  

---

# 1. Burgers Equation: Standard vs Fourier Feature PINNs

## Governing Equation

The viscous Burgers equation:

$$
u_t + u\,u_x = \nu u_{xx}, \quad \nu = \frac{0.01}{\pi}
$$

Domain:

$$
x \in [-1,1], \quad t \in [0,1]
$$

Initial condition:

$$
u(x,0) = -\sin(\pi x)
$$

Boundary conditions:

$$
u(-1,t) = 0, \quad \nu(1,t) = 0
$$

---

## PINN Formulation

A neural network $u_\theta(x,t)$ is trained by minimizing:

$$
\mathcal{L} = \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{IC}} + \mathcal{L}_{\text{BC}}
$$

### PDE Residual Loss

$$
\mathcal{L}_{\text{PDE}} = \mathbb{E}\left[(u_t + u u_x - \nu u_{xx})^2\right]
$$

### Initial Condition Loss

$$
\mathcal{L}_{\text{IC}} = \mathbb{E}\left[(u_\theta(x,0) + \sin(\pi x))^2\right]
$$

### Boundary Condition Loss

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}\left[(u_\theta(\pm1,t))^2\right]
$$

---

## Model Architectures

### Standard PINN

- Fully connected network  
- Architecture: $2 \rightarrow 128 \rightarrow 128 \rightarrow 128 \rightarrow 1$  
- Activation: $\tanh$

---

### Fourier Feature PINN

Input embedding:

$$
\phi(x,t) = [\sin(2\pi B[x,t]), \cos(2\pi B[x,t])]
$$

$$
B \sim \mathcal{N}(0, \sigma^2)
$$

- Feature dimension: 64  
- MLP: $64 \rightarrow 128 \rightarrow 128 \rightarrow 128 \rightarrow 1$

---

## Training Strategy

- Adam optimizer followed by L-BFGS  
- Collocation points:
  - Interior (PDE)
  - Initial ($t=0$)
  - Boundary ($x=\pm1$)

---

## Results

### Standard PINN

- Captures global structure  
- Error concentrated near shock  

$$
|u_t + u u_x - \nu u_{xx}| \text{ increases near } x \approx 0
$$

---

### Fourier Feature PINN

- Better gradient resolution  
- Reduced error near shock  
- Improved spectral representation  

---

## Residual Diagnostic

$$
R(x,t) = \left| u_t + u u_x - \nu u_{xx} \right|
$$

- Standard PINN: localized high residual  
- Fourier PINN: more uniform residual  

---

# 2. Underdamped Harmonic Oscillator PINN

## Governing Equation

$$
\frac{d^2 x}{dz^2} + 2 \xi \frac{dx}{dz} + x = 0
$$

- $z$: time variable  
- $\xi \in [0.1, 0.4]$: damping ratio  

---

## PINN Formulation

The network approximates:

$$
x_\theta(z, \xi)
$$

Loss:

$$
\mathcal{L} = \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{IC}}
$$

### PDE Residual

$$
\mathcal{L}_{\text{PDE}} = \mathbb{E}\left[\left(\frac{d^2 x}{dz^2} + 2\xi \frac{dx}{dz} + x \right)^2\right]
$$

### Initial Conditions

$$
x(0) = 0.7, \quad \frac{dx}{dz}(0) = 1.2
$$

$$
\mathcal{L}_{\text{IC}} = (x(0)-0.7)^2 + \left(\frac{dx}{dz}(0)-1.2\right)^2
$$

---

## Model

- Input: $(z, \xi)$  
- Architecture: $2 \rightarrow 64 \rightarrow 64 \rightarrow 64 \rightarrow 1$  
- Activation: $\tanh$

---

## Training

- Adam optimizer  
- Domain:
  - $z \in [0,20]$
  - $\xi \in [0.1,0.4]$

---

## Results

The learned solution follows:

$$
x(z,\xi) \approx e^{-\xi z} \cos(\omega z), \quad \omega = \sqrt{1-\xi^2}
$$

### Observations

- Lower $\xi$: slower decay, sustained oscillations  
- Higher $\xi$: faster decay, reduced amplitude  

---

# Key Observations

## Burgers Equation

- Standard PINNs under-resolve sharp gradients  
- Fourier features improve high-frequency representation  

---

## Harmonic Oscillator

- Single network learns parametric dependence on $\xi$  
- Captures full solution family  

---

# Conclusion

- PINNs enforce governing equations via residual minimization  
- Automatic differentiation provides exact derivatives  
- Fourier embeddings improve performance for high-frequency regimes  
- Parametric PINNs learn continuous solution manifolds
