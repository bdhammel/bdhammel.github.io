---
layout: post
title:  "The Ising model"
---

This post provides a more detailed discussion of the theory behind [my python routine](https://github.com/bdhammel/ising-model) for simulating phases transition in the Ising model of a ferromagnet.


The Ising Model is a simplified version of a ferromagnet - where the structure of the material consist of a single dipole per lattice site. The overall magnetization of the material is determined by the number of dipoles that are aligned parallel to one-another. The Ising Model is a beautifully simple demonstration of the implications of statistical mechanics and phase transitions - as well a being an fantastic example of the power of Monte Carlo Simulations. This post covers running the metropolis algorithm for the classical Ising model.

We start with a macroscopic state, state '\\( f \\)', of the Ising-Ferromagnet; the bottom half of the material is aligned with each microscopic state in spin-up and the top half is aligned with spin-down, with the exception of one dipole who's spin-up - anti-parallel to it's neighbors. 

![Ising model initial state]({{ site.url}}/assets/ising_model/monte-carlo-ising-2.png)

A microscopic state is selected at random. Lets say we land on the state aligned anti-parallel to it's neighbors.

![Ising model initial state]({{ site.url}}/assets/ising_model/monte-carlo-ising-3.png)

The change in energy of the system is dictated by the interaction of a dipole with its neighbors [eq. 1]. 

$$
\begin{equation}
\Delta E=-2J\sum_{k}s_{l}s_{k}
\end{equation}
$$

If the spin of this dipole were to be reversed: 

![2D Ising model]({{ site.url}}/assets/ising_model/monte-carlo-ising-4.png)

the energy of the system would change by:

$$
\Delta E=-8J
$$

thus creating a new macroscopic state,  state '\\( f' \\)'. Because the change in energy is negative, and we know that energy will be at a minimum for equilibrium, we'll accept the change,  '\\( f = f' \\)': 

we then randomly select another dipole in the system.

![2D Ising model]({{ site.url}}/assets/ising_model/monte-carlo-ising-5.png)

Again, we calculate the change in energy if this state were to flip '\\( f \rightarrow f' \\)'. In this case the energy is positive, \\( \Delta E=8J \\). This is still a possible state for the system to be in, however improbable. So we calculate the ratio of transition probabilities based on the partition function of the system

$$
\begin{equation}
\frac{P(f\rightarrow f')}{P(f' \rightarrow f)}=\frac{e^{-\beta E_{2}}}{e^{-\beta E_{1}}}=e^{-\beta \Delta E}
\end{equation}
$$

We determine this transition by "rolling the dice". If this ratio is greater than a random number, \\( x \\), between 0 and 1 we accept the new state. Otherwise, we reject it, and keep the original state

$$
f=\left\{\begin{matrix}
f' & e^{-\beta \Delta E } \geq x \\ 
f & e^{-\beta \Delta E } <  x
\end{matrix}\right.
$$

In terms of computer code, this is all very simple. It's just a 'for loop' with a few 'if' statements. Attached bellow is some basic psudo code to simulating the model, you can see my full code [here](https://github.com/bdhammel/ising-model).

~~~python
system = build_system()

for _ in range(epochs):

    N, M = generate_random_coordinate_location()

    E = calculate_energy_of_fliped_spin()
    
    if E <= 0.:     # Then this is a more probable state, flip the spin
        system[N,M] *= -1
    elif np.exp(-1./T*E) > np.random.rand():    # Still a possbile state, roll the dice
        system[N,M] *= -1

~~~

This is a simulation of a 150x150 lattice in the high temperature limit, \\( T = 10 \frac{J}{KB} \\). 
with \\( 10^{6} \\) samplings 


<video controls>
  <source src="{{ site.url}}/assets/ising_model/highT.webm" type="video/webm">
  <source src="{{ site.url}}/assets/ising_model/highT.mp4" type="video/mp4">
  Your browser does not support the <code>video</code> element.
</video>

The system remains highly disordered and the net magnetization \\( \approx  0 \\). 

In comparison, this is the same simulation, but now at a low temperature limit, \\( T = 0.1 \frac {J}{KB} \\). As the simulation progresses the interaction between the spins dominates  and  causes alignment. Distinct phases appear in the model.

<video controls>
  <source src="{{ site.url}}/assets/ising_model/lowT.webm" type="video/webm">
  <source src="{{ site.url}}/assets/ising_model/lowT.mp4" type="video/mp4">
  Your browser does not support the <code>video</code> element.
</video>

Given enough time, the system will become fully magnetized. 

We expect there is some temperature at which this phase transition happens - where the systems goes from being a Ferromagent to a Paramagnet. This temperature was solved for exactly by Lars Onsager in 1944. 

$$
\begin{equation}
\frac{K_{b}T_{c}}{J}=\frac{2}{ln(1+\sqrt{2})}\approx 2.269
\end{equation}
$$

His solution, although elegant, is immensely complicated. We're going to use the monte carlo method to see the effects that his solution describes. 

![Magnetization vs Temperature for the 2D Ising model]({{ site.url}}/assets/ising_model/monte-carlo-ising-6.png)

The above graph is the result of running the Ising simulation at incrementing temperatures, and calculating the magnetization. As you can see, the magnetization quickly drops from 1 to 0 right around 2.269 given in eqn. 3 and marked by the red line.

In conjunction, we see the Heat capacity of the system spikes at this same temperature. 

![Heat Capacity vs Temperature for the 2D Ising model]({{ site.url}}/assets/ising_model/monte-carlo-ising-7.png)

In the same way that as water boils on the stove top, it doesn't matter how much energy you pump into the system, the temperature will remain the same during a phase transition.

We've been able to gain a detailed understanding of the behavior of the Ising model through this method, and we've done it without any calculus. All we did was set the boundary conditions of the system, and let the computer carry out the the behavior of nature with good old fashion brute force. I find this pretty incredible. 

*One thing to note:* The results of this simulation (and monte carlo simulations in general) are very depended on the computation time, and the size of the system you work with. In the video showing the low temperature limit we saw distinct phases form in the model. However, it would require a significant amount of time for the model to become completely magnetized in one direction or the other. For the simulations of magnetization and heat capacity: I started with the system in a fully magnetized state, and watched for what temperature it deteriorated. To show the effect run time has on this, bellow is a graph of the magnetization of the system for different sampling numbers. 


![Ising model magnetization for different run times]({{ site.url}}/assets/ising_model/monte-carlo-ising-8.png)

As you can see, it has a big effect on the reliability of the results. In practice, these codes are run on enormous super computers for months at a time to generate trusted results.

![Ising model around the critical temperature]({{ site.url }}/assets/ising_model/monte-carlo-ising-9.png)
