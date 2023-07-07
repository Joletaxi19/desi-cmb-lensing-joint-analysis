limber.py is used to compute Ckg, Cgg. The power spectrum prediction (Pgm, Pgg, Pmm) isn't built in, and is designed to be modular (e.g. can swap out velocileptors for HEFT quite easily). The same is true for the background prediction.

pkCodes.py contains several methods to compute Pgm, Pmm, Pgg

background.py contains several methods to computed background quantities relevant for Limber integration. I'm currently running Joe's HEFT emulator on my laptop (July 7, 2023), which I modified to make it compatible with jax. For now I'm using hard-coded paths to use this emulator, obviously in the future the emulator will hopefully be pip-installable (including the jax-wrapped code that I have?).

limber_jax.py is the most up-to-date code (as of Jul 7 2023). It is written in jax, making it differentiable.
