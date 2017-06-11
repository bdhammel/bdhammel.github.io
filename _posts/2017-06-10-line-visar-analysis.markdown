---
layout: post
title:  "Line-VISAR analysis"
date:   2017-06-10 13:48:19 -0700
---

This post is more-or-less a chapter out of my dissertation. I will note the my dissertation does have a very glaring error in it (in my method of phase unwrapping) - the kind that keeps me awake at night. My initial approach only added a small error (which is why it escaped my attention); Nevertheless, it is --obviously-- wrong. I try and clear my concious by correcting that here...

The code for this analysis can be found here: [https://github.com/bdhammel/line-visar-analysis](https://github.com/bdhammel/line-visar-analysis)

## Velocity Interferometer System for Any Reflector

The Velocity Interferometer System for Any Reflector (VISAR)\cite{Barker:1972ly} is a principal diagnostic in dynamic compression research. The VISAR works by analyzing the Doppler shift of a pulse of light which has been reflected off of a moving surface (e.g. the back surface of a sample as a shock wave emerges). By examining the frequency change of the light, the velocity of the moving surface can be found, allowing the material properties (such as the pressure) to be inferred from the application of classical mechanics including the laws of mass, momentum, and energy conservation, as well as knowledge of the initial physical condition of the target material \cite{Zeldovich:2002cv}.

There are several forms of velocimetry systems, each containing intrinsic benefits and limitations. The work in this dissertation was carried out with a line-imaging VISAR, for its ability to provide spatial information at the target surface. The design (Figure \ref{fig:visar_layout}) was based on the system implemented at the Omega Laser facility \cite{Celliers:2004uo}.  Several other systems (Conventional VISAR \cite{Barker:1972ly}, Push-Pull (Quadrature) VISAR, Heterodyne velocimetry or Photon Doppler Velocimetry) can also be used to probe ultra-fast movements. However these methods have both pros and cons in comparison to the line-VISAR system and should be considered depending on the requirements of the user \cite{Robinson:2005fk, Barker:uq}.

## VISAR Theory of Operation

\begin{figure}[h!]
    \centering
    \includegraphics[width=.85\linewidth]{ch_diag/img/visar/visar_layout.png}
    \caption{Beam path through the VISAR.}
    \label{fig:visar_layout}
\end{figure}

The frequency change in the VISAR's probe beam, due to a moving target with velocity $v(t)$, is governed by the Doppler equation,

$$
\begin{equation}
\lambda (t)=\lambda _{0}\left( 1-\frac{2v(t)}{c} \right ).
\end{equation}
$$

\noindent
The Doppler shifted light is passed through a Mach-Zehnder interferometer (Figure \ref{fig:visar_layout}) where one leg of the interferometer is delayed by placing an etalon in the ray path. The recombination of the two legs generates a fringe comb pattern overlaid on the image of the target surface. A line-out of this is then recorded using a white-light streak camera. The result is an image with spatial and temporal information of the target, as seen in Figure \ref{fig:raw}. The shift in the fringes is directly proportional to the velocity of the object that the light was reflected from, 

$$
v \propto \left \{ \text{fringe shift}\right \}.
$$

\begin{figure}[h!]
    \centering
    \includegraphics[width=.65\linewidth]{ch_diag/img/visar/raw.png}
    \caption{Raw VISAR data recorded by the streak camera. For this work, the spatial width is a 1D line through an illuminated \SI{1.5}{mm} field-of-view on the target back-surface, and the temporal width is \SI{410}{ns}. The increase in reflectivity at the time of breakout is expected to be contributed to an improvement in the target surface quality, such that it becomes smoother and a better reflector during the acceleration of the breakout \cite{Smith:sf}.}
    \label{fig:raw}
\end{figure}

\noindent
The proportionality constant of this relation is found by starting with the equation of a plane wave,

$$
\begin{equation}
\mathbf{E}=\mathbf{E_0}e^{i \left ( \mathbf{k}\cdot\mathbf{r} - \omega t + \delta \right )},
\label{eqn:light}
\end{equation}
$$

\noindent
and solving for the superposition of the beams from both legs of the interferometer at the detector surface \cite{Fowles:1989qv},

$$
\begin{align}
I &\equiv  \left | \mathbf{E} \right | \nonumber \\
&= \mathbf{E}\cdot\mathbf{E}^*  \nonumber \\
&=\left(\mathbf{E_1}+\mathbf{E_2}\right) \cdot \left(\mathbf{E_1^*}+\mathbf{E_2^*}\right) \nonumber \\
%&=\left |\mathbf{E_1}\right|^2+ \left|\mathbf{E_2} \right|^2 + \mathbf{E_{1,0}}\cdot \mathbf{E_{2,0}} \left ( \frac{e^{i \left ( \left (\mathbf{k_1} - \mathbf{k_1} \right ) \cdot\mathbf{r} - \omega_1 t + \delta\right )} + e^{-i \left ( \left (\mathbf{k_1} - \mathbf{k_2}\right ) \cdot\mathbf{r} - \omega_2 t + \delta\right )} }{2} \right )  \nonumber \\
&=I_1+I_2 + 2\mathbf{E_{1,0}}\cdot \mathbf{E_{2,0}}\cos \theta.
\end{align}
$$

\noindent
The $2\mathbf{E_{1,0}}\cdot \mathbf{E_{2,0}}\cos \theta$ term describes the resulting interference. $\theta = \left (\mathbf{k_1} - \mathbf{k_2} \right ) \cdot\mathbf{r} - \omega_1 t + \omega_2 t $ and can be simplified as a time dependent phase with a constant offset, $\phi (t) + \delta_{0}$. Wherein 

$$
\begin{equation}
\phi(t) = 2\pi F(t)
\label{eqn:fshift}
\end{equation}
$$

\noindent
such that $F(t)$ is the fractional fringe shift, proportional to the velocity of the target. The intensities can then be grouped into a background term, $b(t)$, and amplitude of the interference, $a(t)$, for a general form:

$$
\begin{equation}
I(t) = b(t)+a(t)\cos[\phi (t) + \delta_{0}].
\label{eqn:1dgen}
\end{equation}
$$

Considering the 1D case, with the optical system in perfect alignment such that all interferometer optic surfaces are perfectly parallel, the interference of two beams, $\mathbf{E_1}$ and $\mathbf{E_2}$ (eqn. \ref{eqn:light}), will be determined by the time delay, 

$$
\begin{equation}
\tau = \frac{2d}{c}(n-1).
\label{eqn:1ddelay}
\end{equation}
$$

\noindent
In which $\tau$ is a function of the etalon (thickness $d$) placed in the path of one leg of the interferometer. The Doppler-shifted frequency of the reference beam and the time delayed beam can then be used to find $\theta$,

$$
\begin{align*}
\theta(t) &= \left ( \mathbf{k_1}(t_1) - \mathbf{k_2}(t_2) \right ) \cdot\mathbf{r} - \omega_1 t_1 + \omega_2 t_2  \nonumber \\
&\Rightarrow \left (k_1(t+\tau) - k_2(t) \right ) z - \omega_1(t + \tau) \left (t+\tau \right) + \omega_2(t) t \nonumber \\
&= 2 \pi \left \{ \frac{1}{\lambda_0}\left( \frac{c}{c-2v(t+\tau)} \right) - \frac{1}{\lambda_0}\left( \frac{c}{c-2v(t)} \right) \right \} z  \nonumber \\
&- 2 \pi c \left \{ \frac{1}{\lambda_0}\left( \frac{c}{c-2v(t+\tau)} \right)\left( t + \tau \right) - \frac{1}{\lambda_0}\left( \frac{c}{c-2v(t)} \right) t \right \} 
\end{align*}
$$

\noindent 
The result then can be simplified by assuming the surface is stationary at time zero and the k vectors are separated by a distance determined by the time delay of the etalon, $v(t)_{t=0} = 0$ and $ z = c\tau$,

$$
\begin{align*}
%\theta &= \frac{2 \pi}{\lambda_0} \left \{  \overbrace{ \left(  \frac{c}{c-2v(\tau)} - 1 \right) }^{  \frac{c-c+2v(\tau)}{c-v(\tau)}} z - \left( \frac{c}{c-v(\tau)} \right) c \tau \right \} \\
\theta(t) &= \frac{2 \pi}{\lambda_0} \left \{  \left(  \frac{c}{c-2v(\tau)} - 1 \right) z - \left( \frac{c}{c-2v(\tau)} \right) c \tau \right \}  \\
&= \frac{2 \pi}{\lambda_0} \left \{  \left(  \frac{ 2v(\tau)}{c-2v(\tau)} \right) z - \left( \frac{c}{c-2v(\tau)} \right) c \tau \right \}.
\end{align*}
$$

\noindent
Lastly, given that $ v(t) << c$, $\theta$ can be further reduced given that $c - 2v(t) \approx c $,

$$
\begin{align*}
\theta(t) &= \frac{2 \pi}{\lambda_0} \left \{  \left(  \frac{ 2v(\tau)}{c} \right) c \tau - c \tau \right \} \\
&= 2 \pi \frac{2 v(\tau)\tau}{\lambda_0} + \delta_0,
\end{align*}
$$

\noindent
Allowing us to solve for $F(t)$ (eqn. \ref{eqn:fshift}) to get a resulting a Velocity Per Fringe (VPF) shift of: 

$$
\begin{equation}
{\rm VPF} \equiv \frac{v(\tau)}{F(t)} = \frac{\lambda_0}{2\tau}.
\label{eqn:1dvpf}
\end{equation}
$$

Extending this formalism to all space, and taking into consideration the imaging properties of the line-VISAR system, the function of intensity (eqn. \ref{eqn:1dgen}) takes on a form dependent on $x$ and $t$,

$$
\begin{equation}
I(x,t) = b(x,t)+a(x,t)\cos[\phi (x,t)+2\pi f_{0}x + \delta_{0} ].
\label{eqn:2dgen}
\end{equation}
$$

\noindent
The equations for VPF (eqn. \ref{eqn:1dvpf}) and $\tau$ (eqn.\ref{eqn:1ddelay}) are then adjusted to account for dispersion of light in the etalon and the shift in the image location due to the etalon's index of refraction (and movement of $\text{M}_2$, Figure \ref{fig:wl_full}, to compensate for this):

$$
\begin{align}
{\rm VPF} &= \frac{\lambda_0}{2\tau(1+\delta)} \label{eqn:2dvpf}, \\ 
\tau &= \frac{2d}{c}(n-1/n) \label{eqn:2dtau}.
\end{align}
$$

## Analysis

As discussed in the first section of this chapter, the fringe shift recorded is directly proportional to the velocity of the target. It is, therefore, the goal of the analysis routine to extract the percentage fringe shift (at a given location and time). Several methods exist for accomplishing this. However, the Fourier transform method (described below), first proposed by Takeda et al. \cite{M.-Takeda:1983bh}, has been determined to be the most accurate \cite{Celliers:2004uo}.

The raw data (fig. \ref{fig:raw}) is typically cropped to analyze the region of planar shock breakout (Figure \ref{fig:visar_analy}-a). This image can be described mathematically using equation \ref{eqn:2dgen}, with $ b(x,t ) = I_1+I_2 $ being the background and $ a(x,t) =  2\mathbf{E_1}\mathbf{E_2} $ describing the intensity of the fringes. $\phi (x,t)$ represents the phase of the fringes and $2\pi f_{0}x + \delta_{0} $ describes the linear phase ramp of the background fringe pattern. The goal is to find $ \phi (x,t) $, which is directly proportional to the velocity of the target via equation \ref{eqn:fshift} and equation \ref{eqn:2dvpf}. Rewriting equation \ref{eqn:2dgen} in terms of its complex components yields:

$$
\begin{flalign}
&& f(x,t) &= b(x,t)+c(x,t)e^{i2\pi f_{0}x} + c^*(x,t)e^{-i2\pi f_{0}x },& \\
\text{with} && c(x,t) &= \frac{1}{2}a(x,t)e^{i\delta_{0}}e^{i\phi (x,t)}. 
\end{flalign}
$$

\noindent
Applying a Fourier transform to the data at each point-in-time (Figure \ref{fig:visar_analy}-b) allows for the filtering of specific frequencies, such that the background, $b(x,t)$, can be removed by setting the pixel values to zero, as illustrated in Figure \ref{fig:visar_analy}-c and  \ref{fig:visar_analy}-d:

$$
\begin{align*}
F(f,t) &= B(f,t)+\int_{ -\infty }^{\infty} c(x,t) e^{i2\pi f_{0}x}e^{-ifx} \; dx + \int_{-\infty }^{\infty}c^*(x,t)e^{-i2\pi f_{0}x }e^{-ifx} \; dx \\[1.5ex]
&=B(f,t)+\int_{-\infty }^{\infty}c(x,t)e^{i2\pi (f_{0}-f) x} \; dx + \int_{-\infty }^{\infty}c^*(x,t)e^{-i2\pi (f_{0} +f )x} \;dx \\[1.5ex]
&=\cancelto{0}{B(f,t)}+\cancelto{0}{C^*(f+f_0,t)}+C(f-f_0,t).
\end{align*}
$$

\noindent
Applying an inverse Fourier transform (Figure  \ref{fig:visar_analy}-e), this can be written as:

$$
\begin{align}
d(x,t) &= \int_{-\infty}^{\infty}C(f-f_0,t)e^{ixf} \; df \nonumber \\
&= \int_{-\infty}^{\infty}C(f-f_0,t) \left ( \cos(xf) + i\sin(xf) \right ) \; df  \label{eqn:ifft} \\
&=c(x,t)e^{2\pi i f_0x} \label{eqn:filtered}
\end{align}
$$

\noindent
Equation \ref{eqn:filtered} has both a real and imaginary valued functions (as seen in eqn. \ref{eqn:ifft}). Where

$$
\begin{flalign}
&& \operatorname{Re} [d(x,t) ] &\propto  \sin( \phi (x,t) + 2\pi f_0 x + \delta_0) \label{eqn:re},& \\ 
\text{and} && \operatorname{Im} [d(x,t) ] &\propto  \cos( \phi (x,t) + 2\pi f_0x + \delta_0) \label{eqn:im},
\end{flalign}
$$

\noindent
are $\pi/2$ out of phase. Taking the $\arctan$ of the ratio (Figure  \ref{fig:visar_analy}-f) allows the phase, $\phi (x,t) + 2\pi f_0x + \delta_0$, to be extracted: 

$$
\begin{equation*}
W( \phi (x,t) + 2\pi f_0x + \delta_0) = \arctan\left ( \frac{\operatorname{Re} [d(x,t) ]}{\operatorname{Im} [d(x,t) ]} \right ).
\end{equation*}
$$

\noindent
The resulting function $W$ has discontinuities representing $\pi$ shifts as the $\arctan$ moves through full rotations. The velocity signal can be constructed by setting these shifts to zero and scaling the values by the proportionality factor VPF (eqn: \ref{eqn:2dvpf}). The programatic method for reconstructing the velocity trace from the time dependent values in wrapped\_phase parameters can be accomplished via the (Python) script bellow:
\vspace{4mm}
 
\begin{minipage}{\linewidth}
\begin{lstlisting}[language=python] 
_threshold = .07
_max_dphase = np.pi/2. - _threshold
_min_dphase = -1 * _max_dphase
vpf = 1.998 # velocity per fringe shift for a given etalon
 
for i in range(1, len(wrapped_phase)):
    dphase = row[i] - row[i-1] 
    if dphase < _min_dphase:
        dphase = 0
    elif dphase > _max_dphase:
        dphase = 0
    v += dphase * vpf / (2*np.pi)
\end{lstlisting}
\end{minipage}
\vspace{4mm}

\noindent
The \lstinline[language=python]{_threshold} is necessary because of noise in the system but must be conservatively chosen, as not to introduce unwanted errors.  After doing this for each spatial point, the full velocity map can be generated (Figure \ref{fig:visar_analy}-g and one-dimensional line-out \ref{fig:visar_analy}-h):

\begin{figure}[h!]
    \centering
    \includegraphics[width=\linewidth]{ch_diag/img/visar/visar_analy.png}
    \caption{Graphical output of VISAR analysis steps.}
    \label{fig:visar_analy}
\end{figure}
