# Thermal-Team-Full-Mission-Sim

This is a public copy of the full mission simulation that Ekansh Sharma and I, Pratyush Majumdar, have been working on together starting from August 21st. This model is to simulate a novel wooden CubeSat, WolfSat-2, and includes various different models. We use astropy and BoInOr libraries for our code.

1. Altitude Modelling

We gathered data using pymsis, and MSIS data wrapper, to accurately retrieve atmospheric entities for atmospheric drag calculations. We use Cowell Propagation for the Altitude Modelling portion.

2. Temperature Modelling

We used rotation matrices, coordinate transforms, and eclipsing functions to determine whether the satellite is in sunlight or not. We perform steady state calculations and thermal expansion results. Our conclusion is that temperature effects do not significantly affect the structural stability of WolfSat-2, especially given it's short lifetime.

3. Power Modelling

To simulate whether two solar panels is enough for WolfSat 2, we have taken the following approach.
We are to consider all faces to be producing “power” by way of solar panels
We will apply the general power generation equation for solar panels with two additional factors

This power equation defines the power generation model of our mission simulation

P_faces[i] = S_sun * 0.61272 * Lx_panel * Ly_panel * cell_count * n_eff * Sun_cosine[
   i] * sun_state * SOLAR_CONDITION_FACTOR

Where

S_sun = 1361 W/m^2

0.61272 = the proportion of panel area that are actual solar cells

Cell_count = set to 1 for simplicity, treating each face as having only one panel. I do not believe setting this to 3 would change much, as then we would have to change the proportion of area, which would equalize the effects back to just setting cell count to 1

N_eff = solar panel efficiency. Set to 0.20

Sun_cosine = incident angle between sun’s rays and panel surface

Sun_state = a modifier that is either 0 or 1 at any given moment. It is 0 when the satellite is in eclipse, so that we do not count time spent in shadow as actually generating power

SOLAR_ CONDITION_FACTOR = An aggregate for conditions external to satellite function that may further reduce solar input. Set to 0.4 to simulate particularly harsh external conditions

We will implement tumbling along 3 axes for our incidence angle calculations
After the simulation, we would sum up the powers of any two non-touching sides, excluding z faces, and that is the total power generated per orbit in Wh

After testing harsh conditions, with a solar condition modifier of 0.4 and a solar panel efficiency of 0.2, we have concluded a net power generation of 0.3316 to 0.346 Wh per orbit, not factoring in energy storage and other qualifications that may affect how much “real” power is available for WS2 to use.

These factors can be changed to suit your modelling needs.
