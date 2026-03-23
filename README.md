# SHM for Under Damped Regions PINNs

This repository contains two implementations of Physics-Informed Neural Networks (PINNs):

1. Burgers equation with comparison between standard and Fourier feature PINNs  
2. Underdamped harmonic oscillator as a parametric PINN  

The focus is on formulation, training behavior, and quantitative solution quality.

---

# 1. Burgers Equation: Standard vs Fourier Feature PINNs

## Governing Equation

The viscous Burgers equation is given by:

\[
u_t + u\,u_x = \nu u_{xx}, \quad \nu = \frac{0.01}{\pi}
\]

Domain:

\[
x \in [-1,1], \quad t \in [0,1]
\]

Initial condition:

\[
u(x,0) = -\sin(\pi x)
\]

Boundary conditions:

\[
u(-1,t) = 0, \quad \nu(1,t) = 0
\]

---

## PINN Formulation

A neural network \( u_\theta(x,t) \) is trained by minimizing:

\[
\mathcal{L} = \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{IC}} + \mathcal{L}_{\text{BC}}
\]

### PDE Residual Loss

\[
\mathcal{L}_{\text{PDE}} = \mathbb{E}\left[(u_t + u u_x - \nu u_{xx})^2\right]
\]

### Initial Condition Loss

\[
\mathcal{L}_{\text{IC}} = \mathbb{E}\left[(u_\theta(x,0) + \sin(\pi x))^2\right]
\]

### Boundary Condition Loss

\[
\mathcal{L}_{\text{BC}} = \mathbb{E}\left[(u_\theta(\pm1,t))^2\right]
\]

---

## Model Architectures

### Standard PINN

- Fully connected network  
- Architecture: \(2 \to 128 \to 128 \to 128 \to 1\)  
- Activation: \(\tanh\)

---

### Fourier Feature PINN

Input embedding:

\[
\phi(x,t) = \left[\sin(2\pi B[x,t]), \cos(2\pi B[x,t])\right]
\]

where:

\[
B \sim \mathcal{N}(0, \sigma^2)
\]

- Feature dimension: 64  
- MLP: \(64 \to 128 \to 128 \to 128 \to 1\)

---

## Training Strategy

- Stage 1: Adam optimization  
- Stage 2: L-BFGS refinement  
- Sampling:
  - Interior points for PDE residual  
  - Initial condition points at \(t=0\)  
  - Boundary points at \(x = \pm 1\)

---

## Results

### Standard PINN

- Captures global solution behavior  
- Smooth approximation of wave propagation  
- Residual increases near shock formation region  

\[
|u_t + u u_x - \nu u_{xx}| \text{ is high near } x \approx 0
\]

---

### Fourier Feature PINN

- Improved representation of steep gradients  
- Better resolution of high-frequency components  
- Lower residual near shock region  

---

## Residual Diagnostic

\[
R(x,t) = \left| u_t + u u_x - \nu u_{xx} \right|
\]

- Standard PINN: localized high residual near shock  
- Fourier PINN: more uniform residual distribution  

---

# 2. Underdamped Harmonic Oscillator PINN

## Governing Equation

\[
\frac{d^2 x}{dz^2} + 2 \xi \frac{dx}{dz} + x = 0
\]

- \(z\): time variable  
- \(\xi \in [0.1, 0.4]\): damping ratio  

---

## PINN Formulation

The network approximates:

\[
x_\theta(z, \xi)
\]

Total loss:

\[
\mathcal{L} = \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{IC}}
\]

### PDE Residual

\[
\mathcal{L}_{\text{PDE}} = \mathbb{E}\left[\left(\frac{d^2 x}{dz^2} + 2\xi \frac{dx}{dz} + x \right)^2\right]
\]

### Initial Conditions

\[
x(0) = 0.7, \quad \frac{dx}{dz}(0) = 1.2
\]

\[
\mathcal{L}_{\text{IC}} = (x(0)-0.7)^2 + \left(\frac{dx}{dz}(0)-1.2\right)^2
\]

---

## Model

- Input: \((z, \xi)\)  
- Architecture: \(2 \to 64 \to 64 \to 64 \to 1\)  
- Activation: \(\tanh\)

---

## Training

- Optimizer: Adam  
- Domain:
  - \(z \in [0,20]\)  
  - \(\xi \in [0.1,0.4]\)

---

## Results

The network captures underdamped oscillatory behavior:

\[
x(z,\xi) \approx e^{-\xi z} \cos(\omega z), \quad \omega = \sqrt{1-\xi^2}
\]

### Observations

- Decreasing \(\xi\):
  - Slower decay  
  - Sustained oscillations  

- Increasing \(\xi\):
  - Faster decay  
  - Reduced amplitude  

The model generalizes across \(\xi\) and learns a continuous solution family.

---

# Key Observations

## Burgers Equation

- Standard PINNs struggle with sharp gradients  
- Fourier features improve spectral representation  
- Residual distribution becomes more uniform  

---

## Harmonic Oscillator

- Single network captures parametric dependence on \(\xi\)  
- Learns full solution family without explicit solver  

---

# Conclusion

- PINNs solve nonlinear PDEs and parametric ODEs using residual minimization  
- Automatic differentiation provides exact derivatives  
- Representation choice affects accuracy in high-frequency regimes  
- Fourier embeddings improve performance for problems with sharp features  
- Parametric PINNs enable continuous solution spaces across system parameters
