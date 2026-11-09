# knot-analysis
Python scripts to (a) compute the Alexander polynomial for open/closed chains based on algebraic closure, (b) track the motion of a single knot on a polymer chain simulation, and (c) track two knots.

1. alexpoly_code.py --> This script computes the Alexander polynomial for a single polymer snapshot (xyz coordinates) of a given length. We do not compute the full symbolic polynomial, but a numerical version developed by Peter Virnau and co-workers at the University of Mainz.

2. singleknot_alexpoly_eval_code.py --> This script computes the polynomial for every frame of a trajectory file, typically from a simulation.

3. singleknot_tracking_code.py --> If we know what knot exists on the polymer chain in the simulation, this script can delineate and track the position of this knot over the course of the simulation trajectory.

4. multiknot_tracking_twotref.py --> If we have a simulation with two trefoils (or any two knots of the same type), we can track the position of both knots over the simulation trajectory.

5. multiknot_tracking_trefcinq.py --> If we have a simulation with a trefoil and a cinquefoil (or any two dissimilar knots), we can track the position of both knots over the simulation trajectory. 
