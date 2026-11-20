## Imports
# astropy
from astropy import units as u
from astropy.time import Time
from astropy.time import TimeDelta
from astropy.coordinates import (
    GCRS,
    ITRS,
    CartesianRepresentation,
    CartesianDifferential,
    EarthLocation,
    get_sun,
)
from astropy.coordinates.builtin_frames.intermediate_rotation_transforms import (
    cirs_to_itrs_mat,
    gcrs_to_cirs_mat,
)
from astropy.coordinates.matrix_utilities import matrix_transpose
from astropy.coordinates import get_body_barycentric_posvel

# BoInOr Imports: Blanket 'import BoInOr is rarely used'
from boinor.bodies import Earth
from boinor.twobody import Orbit
from boinor.plotting import OrbitPlotter
from boinor.plotting.orbit.backends import Plotly3D
from boinor.core.propagation import func_twobody
from boinor.twobody.propagation import CowellPropagator
from boinor.ephem import Ephem, EpochsArray
from boinor.bodies import Earth
from boinor.twobody.events import LithobrakeEvent
from boinor.bodies import Earth, Sun
# MSIS
import pymsis
from pymsis import msis
# PPIGRF (Pure Python International Geomagnetic Reference Field)
import ppigrf
from skyfield.api import wgs84, load

# Skyfield (supposedly this makes coord transforms easy)
import skyfield
# Other
import math
import numpy as np
#from scipy.special import erf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
# Import interpolate function
from scipy.interpolate import interp1d

import warnings
from astropy.utils.exceptions import AstropyWarning

warnings.filterwarnings(
    'ignore',
    message='Tried to get polar motions for times after IERS data is valid.*',
    category=AstropyWarning
)

#warnings.filterwarnings("ignore", category=AstropyWarning)

### Initialization of Main Constants ###
## Standard Gravitational Parameter of Earth
mu = Earth.k.to(u.km ** 3 / u.s ** 2).value  # GM

# Earth's rotation rate
omega_E = 7.2921159e-5  # rad/s

##Assumed MSIS Parameters: notes ripped straight from the Fluence model. -- Pratyush
# The epoch (2027-12-01) is a future/synthetic date relative
# to CelesTrak's observed F10.7/Ap archive, so auto-download has no real
# observation to return and may silently fall back to climatological
# defaults or an unrelated cached value -- either of which can swing the
# AO density (and hence fluence) by a large factor, since AO production
# scales strongly with solar activity (paper Section 3.1.1).
# Values below correspond to moderate solar activity (F10.7 ~ 150 sfu,
# Ap ~ 15) -- adjust to whatever scenario you intend to model
# (e.g. solar min ~70, solar max ~200+).
F107_ASSUMED = 70.0
F107A_ASSUMED = 70.0
AP_ASSUMED = 15.0

## Epoch Launch Time
epoch = Time("2027-12-01 00:00:00", scale="utc")

# Injection Position
lat_Injection = 42.1 * u.deg  # Latitude of Hokkaido
# Inclination gives the max and min latitude values: N 42.1 to S 42.1.
# While Hokkaido is at N 43.1, we're using 42.1 because that is the inclination given by Interstellar Technologies
# Longtitude of Hokkaido
long_Injection = 142.32 * u.deg
h_0 = 230.0 * u.km

## Orbital Parameters.
# Semi-major Axis
R_E = Earth.R.to(u.km)
a = R_E + h_0

# Inclination
inclination = 42.1 * u.deg

# Eccentricity
e = 0 * u.one  # Our orbit is approximately circular

### Constants ###
# Solar Absorptivity / IR Emissivity
alpha = epsilon = 0.6

# Density of Birch
rho_birch = 650  # kg m^-3

# Specific Heat of Birch
c_birch = 1300  # J kg^-1 K^-1

# Stefan-Boltzmann Constant
sigma = 5.670374419e-8  # (W m^-2 K^-)4

# To modify solar intake. This essentially sums up any conditions external to the solar panel that could reduce solar input, summarized into one factor representaton.
SOLAR_CONDITION_FACTOR = 0.75

## CubeSat properties
# Area of panels
Lx_panel = Ly_panel = 0.1  # m
# Thickness
Lz_panel = 0.006  # m

# Cross-sectional area of the satellite (Panel Area)
A = Lx_panel * Ly_panel  # m^2

# Coefficient of drag
C_D = 2.2  # dimensionless

# Mass of the satellite
m = 1.33  # kg

# Mass of one panel
m_panel = rho_birch * A * Lz_panel

# Placement stuff for solar panels
cell_count = 1
n_eff = 0.20  # Efficiency of solar panel

# Sunlit Fraction
sun_fraction = 0.65  # 65% sunlight per orbit

## Other Important Spin-related Constants
# Principal Moments of Inertia

# I changed this to the tensor form because I thought that was the issue




I_tensor = np.array([
    [1.581e6, -11927.909, -20665.485],
    [-11927.909, 1.659e6, -9781.097],
    [-20665.485, -9781.097, 1.831e6]
]) * 1e-9  # Converts to kgm^2

### Torque changed according to the drag force, so it is inside the propagation loop

### Theta and Omega are initialized inside the propagation loop

# We Consider the Initial Angular Acceleration to be 0 since the only external forces in motion after we launch is atmospheric drag


"""
Here are some values that we might want to change, but it's not required to do so.
If we were to change them, sure, we'd get slightly more accurate results, but if it's a hassle to do so, we might not, because the accuracy change is almost minimal.
"""
# Orbital Period (s)
P_orbit = 90 * 60
# Solar Constant
S_sun = 1361  # W m^-2
# Albedo Flux
albedo = 150  # W m^-2
# Earth IR Flux
IR_earth = 237  # W m^-2


# RAAN
def getRAAN(lat_Injection, long_Injection, h_0, epoch, mu, inclination):
    ## Finding Position Vector
    # Location relative to Earth's center (ECEF) - WGS84 ellipsoid model (astropy does this automatically)
    pointing_location = EarthLocation.from_geodetic(
        lat=lat_Injection,
        lon=long_Injection,
        height=h_0
    )

    # Convert to GCRS Frame
    gcrs_vector = pointing_location.get_gcrs(obstime=epoch)

    # GCRS Position Vector:
    x_gcrs, y_gcrs, z_gcrs = gcrs_vector.cartesian.x.to(u.km).value, gcrs_vector.cartesian.y.to(
        u.km).value, gcrs_vector.cartesian.z.to(u.km).value
    r_mag = np.sqrt(x_gcrs ** 2 + y_gcrs ** 2 + z_gcrs ** 2)

    r_vec = np.array([x_gcrs, y_gcrs, z_gcrs])

    ## Finding Velocity Vector
    # Scalar speed (Magnitude of Velocity vector)
    v_mag = np.sqrt(mu / r_mag)  # r_magnitude = distance from the center = semi-major axis

    # Converting inclination to radians (Python only uses radians for trigonometric functions)
    inc_rad = np.radians(inclination.value)

    # Calculate Velocity Components
    # Derivation for these equations is in the Wolfsat-2 Thermal Subsytem Folder
    v_z = v_mag * np.sqrt((x_gcrs ** 2 + y_gcrs ** 2) / (r_mag ** 2) - np.cos(inc_rad) ** 2)

    h_z = r_mag * v_mag * np.cos(inc_rad)  # Setting the specific angular momentum's z-component to make the math easier

    v_x = (-y_gcrs * h_z - x_gcrs * z_gcrs * v_z) / (x_gcrs ** 2 + y_gcrs ** 2)
    v_y = (x_gcrs * h_z - y_gcrs * z_gcrs * v_z) / (x_gcrs ** 2 + y_gcrs ** 2)

    v_vec = np.array([v_x, v_y, v_z])

    ## Calculate RAAN
    # r_vector
    # v_vector

    # Angular Momentum Vector
    h_vec = np.cross(r_vec, v_vec)

    # Node Vector (Line of Nodes)
    k_vec = np.array([0, 0, 1])  # Earth's North Pole Vector

    n_vec = np.cross(k_vec, h_vec)
    n_x = n_vec[0]
    n_y = n_vec[1]
    n_z = n_vec[
        2]  # This is always 0, but I like when my math is complete, so I will be including it into the magnitude equation :)

    n_mag = np.sqrt(n_x ** 2 + n_y ** 2 + n_z ** 2)

    # RAAN
    RAAN_raw = np.arccos(n_x / n_mag)

    # Logic check (arccos only outputs values for 0 to pi radians)
    if n_y < 0:
        RAAN = 2 * np.pi - RAAN_raw
    else:
        RAAN = RAAN_raw

    # Convert to degrees
    RAAN = np.degrees(RAAN) * u.deg

    print(f"Calculated RAAN: {RAAN:.4f}")
    return (RAAN, r_vec, v_vec, n_vec, r_mag, v_mag, n_mag)


# For a nearly-circular orbit (Eccentricity is roughly = 0), the Argument of Periapsis is considered to be 0.
arg_periapsis = 0 * u.deg

# True Anomaly
def getTRUE_ANOMALY(n_vec, r_vec, n_mag, r_mag):
    # n_vec
    # r_vec

    # True Anomaly
    true_anom_raw = np.arccos(np.dot(n_vec, r_vec) / (n_mag * r_mag))

    # Logic Check
    if r_vec[2] >= 0:  # z
        true_anom = true_anom_raw
    else:
        true_anom = 2 * np.pi - true_anom_raw

    # Convert to degrees
    true_anom = np.degrees(true_anom) * u.deg
    return true_anom


## Functions for Orbital Parameters
RAAN, r_vec, v_vec, n_vec, r_mag, v_mag, n_mag = getRAAN(lat_Injection, long_Injection, h_0, epoch, mu, inclination)
True_Anomaly = getTRUE_ANOMALY(n_vec, r_vec, n_mag, r_mag)

# Semi-Major Axis:
a = a
# Eccentricity:
ecc = e
# Inclination
inc = inclination
# Right Ascension of the Ascending Node (RAAN)
raan = RAAN
# Argument of Periapsis
arg_p = arg_periapsis
# True Anomaly
nu = True_Anomaly

orbit = Orbit.from_classical(Earth, a, ecc, inc, raan, arg_p, nu, epoch=epoch)
print(orbit)

# Test time values: over 1 orbit, 5000 points
days = 1

tofs = TimeDelta(
    np.linspace(
        0 * u.s,
        1440 * u.min,
        5000
    )
)

LR_val = Earth.R.to(u.km).value

lithobrake_event = LithobrakeEvent(LR_val)
events = [lithobrake_event]
# This pauses the simulation if we go below a certain altitude

epochs = orbit.epoch + tofs

##Formulate a grid of MSIS values by altitude, lon, and lat
pre_run_alts = np.linspace(100, 230, 300)
pre_run_lons = np.linspace(-180, 180, 12)
pre_run_lats = np.linspace(-90, 90, 12)

msis_grid = msis.calculate(
    dates=[epoch.datetime],
    lons=pre_run_lons,
    lats=pre_run_lats,
    alts=pre_run_alts,
    f107s=[F107_ASSUMED],
    f107as=[F107A_ASSUMED],
    aps=[[AP_ASSUMED] * 7]
)

# Pull total mass density (index 0)
rho_raw = msis_grid[:, :, :, :, 0]
# Avg rho values across dates (axis 0), lon (axis 1), lat (axis2)
rho_avg_kg_m3 = np.mean(rho_raw, axis=(0, 1, 2))
# Convert kg/m^3 to kg/km^3
rho_avg = rho_avg_kg_m3 * 1e9
# We'll do log-linear interpolation because it is the closest to the exponential nature of rho
log_rho_avg = np.log(rho_avg)
get_log_rho = interp1d(
    pre_run_alts,
    log_rho_avg,
    kind='linear',
    fill_value="extrapolate"
)

# Repeat the same process for Atomic Oxygen (not present yet cuz the AO fluence part isn't implemented yet)
AO_val_raw = msis_grid[:, :, :, :, 2]  # in atoms /m^3
AO_avg_atoms_m3 = np.mean(AO_val_raw, axis=(0, 1, 2))
# Log-linear interpolation
log_AO_am3 = np.log(AO_avg_atoms_m3)
get_log_AO_am3 = interp1d(
    pre_run_alts,
    log_AO_am3,
    kind='linear',
    fill_value="extrapolate"
)

# Creation of empty lists for helpful byproducts
history_times = []
history_ao_densities = []
history_v_rel = []
drag_accel_list = []
state_list = []
sun_state_list = []


def atmospheric_drag_msis(t0, state):
    # Convert Cowell's elapsed time into an actual Astropy time
    current_time = epoch + t0 * u.s

    # Extract position from six-element state vector
    x, y, z = state[0:3]
    gcrs_coords = np.array([x, y, z])
    r_mag = np.linalg.norm(gcrs_coords)
    alt_km = r_mag - Earth.R.to(u.km).value

    # Call interp1d object for rho
    log_rho = get_log_rho(alt_km)
    rho = np.exp(log_rho)

    # While we're here, we'll also get AO stuff as a byproduct so we don't have to do the vector math later
    log_ao = get_log_AO_am3(alt_km)
    ao_num_density = np.exp(log_ao)

    # Extract spacecraft velocity
    vx, vy, vz = state[3:6]

    # Atmospheric velocity due to Earth's rotation
    gcrs_coords = np.array([x, y, z])
    omega_vec = np.array([0, 0, omega_E])

    v_atm = np.cross(omega_vec, gcrs_coords)

    # spacecraft velocity
    v_sc = np.array([vx, vy, vz])

    # Relative velocity of spacecraft with respect to atmosphere
    v_rel = v_sc - v_atm

    # Log helpful byproducts
    history_times.append(t0)
    history_ao_densities.append(ao_num_density)
    history_v_rel.append(v_rel)

    # Cross sectional area in km^2
    A_km2 = A * 1e-6  # m^2 to km^2

    # Relative velocity magnitude
    v_rel_mag = np.linalg.norm(v_rel)

    # drag calculation
    a_drag = (
            -0.5
            * C_D
            * (A_km2 / m)
            * rho
            * v_rel_mag
            * v_rel
    )

    ax, ay, az = a_drag

    return t0, a_drag, state


def f(t0, state, k):
    # Two-body gravitational acceleration
    du_kep = func_twobody(t0, state, k)

    # Atmospheric drag from MSIS
    t0, a_drag, state = atmospheric_drag_msis(t0, state)

    # Appending Lists
    drag_accel_list.append(a_drag)
    state_list.append(state)

    # Put drag acceleration into the 6-element state derivative
    ax, ay, az = a_drag
    du_drag = np.array([
        0,
        0,
        0,
        ax,
        ay,
        az
    ])

    # Total derivative
    return du_kep + du_drag


rr, vv = orbit.to_ephem(
    EpochsArray(
        orbit.epoch + tofs,
        # IMPORTANT! If ur trying to restore the optimization to what it was before, remove 'events' from this function
        method=CowellPropagator(f=f, events=events)
    ),
).rv()


### Outputs in Kilometres

## Since Atomic Oxide can be modeled completely separately, that is what we will be doing.

### Spin and Thermal Implementation:
## Eclipse Function
def eclipser(t, gcrs_coords, R):
    ### This is for spin + eclipser
    ## Sun vectors
    # Sun direction vector (this is what controls the "spinning": important for spinning)
    current_time = epoch + t * u.s
    # Get sun position in GCRS (Earth center is  0,0,0). This is good because "state" array also has the same convention
    sun_coords = get_sun(current_time)
    sun_inertial = sun_coords.cartesian.xyz.to(u.km).value
    sun_direction = sun_inertial - gcrs_coords  # This changes the direction from the earth to the sun, to the satellite to the sun

    sun_vector = R.T @ sun_direction
    sun_vector = sun_vector / np.linalg.norm(sun_vector)  # Dividing sun_vector by the magnitude

    ## Earth vectors
    # Earth direction vector (this is also what controls "spinning": important for spinning)
    # Get current satellite positon

    earth_direction = gcrs_coords.copy()  # Points from Earth's centre to the satellite
    earth_direction *= -1  # Points from the satellite to the Earth's centre

    earth_vector = R.T @ earth_direction
    earth_vector /= np.linalg.norm(earth_vector)  # Dividing earth_vector by the magnitude

    ### Eclipser Only
    """
    This function finds out the eclipse for the satellite at a certain position.
    - We create a line segment from the sun to the satellite (Line Segment, not a line or ray)
    - We create a sphere with the radius of the earth
    - We check for an intersection between that line segment and the sphere:
        - If there is an intersection, we return a 0 (the satellite is in eclipse)
        - If there is not an intersection, we return a 1 (the satellite is basking in the sun)
    That's it
    """
    # Earth Initialization
    earth_pos = [0, 0, 0]
    # We defined R_E before hand
    R_E_km = R_E.to(u.km).value  # (just in case)

    # Sun initialization (we do NOT need to know the radius of the Sun)
    # While technically, yes, there could be a little error with the centre of the sun being eclipsed by the Earth and the edge of the sun still shining light,
    # it's very timely to implement under our time constraints, and would not produce substantial results.
    sun_pos = sun_inertial

    # Satellite initialization at this instant
    sat_pos = gcrs_coords

    """
    r *dot* r = R_E^2 for a sphere centered at the origin (Earth)
    The line segment equation is: r = sat_pos + lambda(sun_pos - sat_pos) # 0 <= lambda <= 1; 1 = sun_pos, 0 = sat_pos
    So substituting, we get (sat_pos + lambda(sun_pos - sat_pos)) *dot* (sat_pos + lambda(sun_pos - sat_pos)) = R_E^2
    Let's let d = sun_pos - sat_pos for simpler math:
    Expanding: (d * d)(lambda)^2 + 2(sat_pos​ * d)lambda + ((sat_pos​ * sat_pos) ​− R_E^2) = 0           (* = dot product since these are vectors)
    Now, we just have to solve for lambda here and check whether it's value is between 0 and 1.

    I will refer to lambda as lam in this code.
    """

    dist = sun_pos - sat_pos

    # From the expanded equation above, we'll define the variables needed for the quadratic equation in the form of ax^2 + bx + c = 0

    a_quad = np.dot(dist, dist)
    b_quad = 2 * np.dot(sat_pos, dist)
    c_quad = np.dot(sat_pos, sat_pos) - R_E_km ** 2

    # Discriminant:
    disc = b_quad ** 2 - 4 * a_quad * c_quad

    if disc < 0:
        sun_state = 1  # Sunlit
    else:
        # Solving for the two lambdas
        lam1 = -1 * (b_quad + np.sqrt(disc)) / (2 * a_quad)
        lam2 = -1 * (b_quad - np.sqrt(disc)) / (2 * a_quad)

        if (0 <= lam1 <= 1) or (0 <= lam2 <= 1):
            sun_state = 0  # Eclipse
        else:
            sun_state = 1  # Sunlit

    return sun_vector, earth_vector, sun_state


## Calculating Spin
def euler_eq(t, omega, I_tensor, Tau_vec):  # t doesn't matter, btw

    omega = np.array(omega)
    Tau_vec = np.array(Tau_vec)

    ## Euler Equation ##
    # I(alpha) + omega *cross* I(omega) = Tau_ext
    inertia_omega = I_tensor @ omega
    cross_term = np.cross(omega, inertia_omega)

    alpha = np.linalg.solve(I_tensor, Tau_vec - cross_term)

    return alpha


## 4th Order Runge-Kutta Method (RK4) Implementations (helps in omega concentration)
# Starting with Quaternion math: (multiplying quaternions)
def quat_multiply(q1, q2):
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2

    a = a1*a2 - b1*b2 - c1*c2 - d1*d2
    b = a1*b2 + b1*a2 + c1*d2 - d1*c2
    c = a1*c2 - b1*d2 + c1*a2 + d1*b2
    d = a1*d2 + b1*c2 - c1*b2 + d1*a2

    multiplied = np.array([a, b, c, d])

    return multiplied

def quat_normalize(q1):
    normalized = q1 / np.linalg.norm(q1)
    return normalized

# Spin Derivatives
def spin_eq(t, spin_current_state, I_tensor, Tau_vec):

    quat = spin_current_state[0:4]
    omega = spin_current_state[4:7]

    omega_quat = np.array([0, omega[0], omega[1], omega[2]])

    # In order to advance a quaternion, we multiply the initial quaternion
    dq_dt = 0.5 * quat_multiply(
        quat,
        omega_quat
    )

    domega_dt = euler_eq(t,
                  omega,
                  I_tensor,
                  Tau_vec
                  )

    dState_dt = np.concatenate((dq_dt, domega_dt))

    return dState_dt



def rk4_spin(t, spin_current_state, dt, I_tensor, Tau_vec):
    # I'll be spacing these out for readability
    k1 = spin_eq(t,
                  spin_current_state,
                  I_tensor,
                  Tau_vec
                  )
    k2 = spin_eq(t + dt / 2,
                  spin_current_state + (dt * k1 / 2),
                  I_tensor,
                  Tau_vec
                  )
    k3 = spin_eq(t + dt / 2,
                  spin_current_state + (dt * k2 / 2),
                  I_tensor,
                  Tau_vec
                  )
    k4 = spin_eq(t + dt,
                  spin_current_state + (dt * k3),
                  I_tensor,
                  Tau_vec
                  )

    dspin_current_state = (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    next_state = spin_current_state + dspin_current_state

    # Normalizing the quaternion so that its magnitude is always 1
    next_state[0:4] = quat_normalize(next_state[0:4])

    return next_state

# Building our rotation matrix using a quaternion (like R from theta)
def quat_to_R(quat):

    # Breaking down the respective components
    w, x, y, z = quat

    # Precalculating some terms for efficiency
    x2, y2, z2, = x**2, y**2, z**2
    wx, wy, wz = w * x, w * y, w * z
    xy, yz, xz = x * y, y * z, x * z

    # Rotation Matrix
    R = np.array([
        [1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)]
    ])

    ### OUR CODE CONVENTION USES R = BODY -> INERTIAL
    ### THEREFORE, R^T = INERTIAL -> BODY

    return R

# This is to make our lives easier for the initial conditions. Calculating a quaternion that's magnitude is 1 is a pain.
# So we're creating a conversion for theta -> quaternions

"""
def xyz_to_quat(theta):
    theta_x, theta_y, theta_z = theta  # angle [rad]

    ####### I'll be honest, this is kinda difficult to grasp my head around, so this is something I'll work on later.
    ####### For now, 

    return quat
"""


def spin_sun_earth(t, quat, gcrs_coords):
    ### Implementation of Spin ###
    # Face normals (orientation of each face)
    face_normals = np.array([
        [1, 0, 0],  # +X face (0)
        [-1, 0, 0],  # -X face (1)
        [0, 1, 0],  # +Y face (2)
        [0, -1, 0],  # -Y face (3)
        [0, 0, 1],  # +Z face (4)
        [0, 0, -1],  # -Z face (5)
    ])

    R = quat_to_R(quat)

    ### Sunlight/eclipse state
    sun_vector, earth_vector, sun_state = eclipser(t, gcrs_coords, R)

    ## Values for Spin
    # Cosine value for each of the panels
    Sun_cosine = face_normals @ sun_vector

    # So that we don't have "negative" values
    Sun_cosine = np.maximum(Sun_cosine, 0.0)

    # Cosine values for each of the panels
    Earth_cosine = face_normals @ earth_vector

    # So that we don't have "negative" values
    Earth_cosine = np.maximum(Earth_cosine, 0.0)

    return Earth_cosine, Sun_cosine, sun_state, R

def unit_mag_vector(t0, state, R):
    # Convert Cowell's elapsed time into an actual Astropy time
    current_time = epoch + t0 * u.s

    # Extract position from six-element state vector
    x, y, z = state[0:3]
    gcrs_coords = np.array([x, y, z])
    r_mag_now = np.linalg.norm(gcrs_coords)

    # In order to convert to ITRS, we need to convert GCRS to astropy terms with units and time attached
    gcrs_astropy = GCRS(
        CartesianRepresentation(
            x * u.km,
            y * u.km,
            z * u.km,
        ),
        obstime=current_time
    )

    # Convert GCRS to ITRS
    itrs_astropy = gcrs_astropy.transform_to(
        ITRS(obstime=current_time)
    )

    ## Convert to Earth coords (latitude, longitude, altitude)
    location_now = EarthLocation(
        x=itrs_astropy.x,
        y=itrs_astropy.y,
        z=itrs_astropy.z
    )

    lat = location_now.lat  # This is probably already in degrees
    lon = location_now.lon
    altitude = location_now.height

    # colatitude
    colatitude_now = 90 * u.degree - lat

    # Returns radial, south, east -- so radial, colatitude, and longitudinal comps
    B_r, B_theta, B_phi = ppigrf.igrf_gc(
        r=r_mag_now, # Already in km
        theta=colatitude_now.to_value(u.deg),
        phi=lon.to_value(u.deg),
        date=current_time.datetime
    )

    # Converting from nanoteslas to teslas
    B_r *= 1e-9
    B_theta *= 1e-9
    B_phi *= 1e-9

    # Numpy expects radians
    colat_rad = math.radians(colatitude_now.value)
    lon_rad = math.radians(lon.value)

    B_spherical = np.array([B_r, B_theta, B_phi]) # In Teslas

    R_spherical_to_ECEF = np.array([
        [np.sin(colat_rad) * np.cos(lon_rad), np.cos(colat_rad) * np.cos(lon_rad), -1 * np.sin(lon_rad)],
        [np.sin(colat_rad) * np.sin(lon_rad), np.cos(colat_rad) * np.sin(lon_rad), np.cos(lon_rad)],
        [np.cos(colat_rad), -1 * np.sin(colat_rad), 0]
    ])

    B_ecef = R_spherical_to_ECEF @ B_spherical
    B_ecef = B_ecef.flatten()


    # 1. Get the constituent rotation matrices for the given time
    # cirs_to_itrs_mat maps CIRS -> ITRS, so its transpose maps ITRS -> CIRS
    mat_itrs_to_cirs = matrix_transpose(cirs_to_itrs_mat(current_time))

    # gcrs_to_cirs_mat maps GCRS -> CIRS, so its transpose maps CIRS -> GCRS
    mat_cirs_to_gcrs = matrix_transpose(gcrs_to_cirs_mat(current_time))

    # 2. Chain them together to get the full ITRS -> GCRS rotation matrix
    matrix_itrs_to_gcrs = mat_cirs_to_gcrs @ mat_itrs_to_cirs

    # 3. Now matrix multiplication
    B_gcrs = matrix_itrs_to_gcrs @ B_ecef

    B_body = R.T @ B_gcrs

    u_mag_vector = B_body / np.linalg.norm(B_body)

    return np.array(B_body), u_mag_vector

"""
def magnetic_torque(t, state, R):
    # Running unit_mag_vector to get our magnetic field values:
    B_body, u_mag_vector = unit_mag_vector(t, state, R)

    ## Magnetic Dipole Moment of magnet (From Bill of Materials)
    m_mag = 0.2  # Amps-m^2

    # +Z face direction vector (no solar panel on this face) (face 4)
    z_face = np.array([0, 0, 1])

    m_vec_body = m_mag * z_face

    # Magnetic torque = m *cross* B
    Tau_magnet = np.cross(m_vec_body, B_body)

    # Alignment angle
    alignment_angle = math.degrees(math.acos(np.clip(
        np.dot(m_vec_body, B_body) / (np.linalg.norm(m_vec_body) * np.linalg.norm(B_body))))
    )


    # Debugging
    '''
    print("B =", B_body)
    print("B magnitude =", np.linalg.norm(B_body), "and unit vector = ", u_mag_vector)
    '''

    return Tau_magnet, alignment_angle, u_mag_vector
"""

# A REALLY rough approximation of damping, where say tau_damp = -k * omega
def damper_torque(spin_current_state):
    # spin_current_state typically contains [omega_x, omega_y, omega_z]
    # (or combined with quaternions depending on your state vector layout)
    omega = spin_current_state[4:7]  # Assuming angular velocity is the first 3 elements

    # Lowball scalar damping coefficient
    k_damp = 1e-4

    # Simple proportional damping torque opposing the rotation
    Tau_damp = -k_damp * omega

    return Tau_damp

def magnetic_torque(t, state, R):
    # Running unit_mag_vector to get our magnetic field values:
    B_body, u_mag_vector = unit_mag_vector(t, state, R)

    ## Magnetic Dipole Moment of magnet (From Bill of Materials)
    m_mag = 0.2  # Amps-m^2

    # +Z face direction vector (no solar panel on this face) (face 4)
    z_face = np.array([0, 0, 1])

    m_vec_body = m_mag * z_face

    # Magnetic torque = m *cross* B
    Tau_magnet = np.cross(m_vec_body, B_body)

    # Alignment angle
    cos_angle = np.dot(m_vec_body, B_body) / (
            np.linalg.norm(m_vec_body) * np.linalg.norm(B_body)
    )
    alignment_angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))


    # Debugging
    '''
    print("B =", B_body)
    print("B magnitude =", np.linalg.norm(B_body), "and unit vector = ", u_mag_vector)
    '''

    return Tau_magnet, alignment_angle, u_mag_vector

### Functions
def derivatives(t, T_faces, T_internal, full_current_state, Earth_cosine, Sun_cosine, sun_state):
    gcrs_coords = full_current_state[0:3]

    # Arrays for the 6 faces
    dT_faces = np.zeros(6)
    P_faces = np.zeros(6)

    ### Conduction stuff ###
    # Nearby faces list for conduction (What faces touch each other)
    Nearby_faces = {
        0: [2, 3, 4, 5],
        1: [2, 3, 4, 5],
        2: [0, 1, 4, 5],
        3: [0, 1, 4, 5],
        4: [0, 1, 2, 3],
        5: [0, 1, 2, 3],
    }

    # Thermal conductance between faces (W K^-1)
    K_conduction = 0.10

    # Thermal conductance - internal components (estimated) (W K^-1)
    K_internal = 0.10

    # Specific Heat of internal components (estimated) (J K^-1)
    C_internal = 500

    # Specific Heat of external components (J K^-1)
    C_external = m_panel * c_birch

    # Loop for each of the 6 panels
    for i in range(6):
        # Heat inputs (Q) for face "i"
        Q_sun = alpha * S_sun * A * (sun_state * Sun_cosine[i])
        Q_albedo = alpha * albedo * A * Earth_cosine[i] * sun_state  # We'll consider sun_state on for now
        Q_IR = epsilon * IR_earth * A * Earth_cosine[i]  # Always present, so no "sun_state"
        Q_radiation = epsilon * sigma * A * (T_faces[i] ** 4)

        # Conduction between the panels
        Q_conduction = 0.0
        for j in Nearby_faces[i]:
            Q_conduction += K_conduction * (T_faces[j] - T_faces[i])

        # Internal conduction
        Q_internal_conduction = K_internal * (T_internal - T_faces[i])

        # Net Heat
        Q_net = Q_sun + Q_albedo + Q_IR + Q_conduction + Q_internal_conduction - Q_radiation

        # dT/dt for external (chassis)
        dT_faces[i] = Q_net / C_external

    # Battery internal heat (W)

    Q_internal_node = 3.0  # Watts, total internal heat (electronics)

    if sun_state == 1.0:
        Q_internal_node += 0.5  # charging heat (battery loss) (just estimating)
    else:
        Q_internal_node += 0.2  # discharging heat (also battery loss) (just estimating)

    # Internal Node Derivative
    Q_internal_total = 0.0
    for i in range(6):
        Q_internal_total += K_internal * (T_faces[i] - T_internal)

    # Adding internal battery heat
    Q_internal_total += Q_internal_node

    dTi_dt = Q_internal_total / C_internal

    # Power generation stuff
    # Loop for each face
    for i in range(6):
        P_faces[i] = S_sun * 0.61272 * Lx_panel * Ly_panel * cell_count * n_eff * Sun_cosine[
            i] * sun_state * SOLAR_CONDITION_FACTOR

    return dT_faces, dTi_dt, P_faces


## Propagation for Spin + Thermal + Power now too! (if it's possible, it may be best to implement these into the f function for optimization)
def propagation(tofs, rr, vv):
    ## Setup variables
    # Initial Conditions
    # Quaternion (Initial angle)
    quat_0 = np.array([1, 0, 0, 0])
    quat = quat_0.copy()

    # Omega (Initial angular velocity)
    omega_0 = np.array([0.05, 0.1, 0.1])
    omega = omega_0.copy()

    # Initial Temperature of External Chassis faces (K)
    T_faces = np.ones(6) * 280  # Considering 280 K for now. Probably gonna change it

    # Initial Temperature of internal environment (K)
    T_internal = 280

    # Lists
    T_ext_list = []
    T_int_list = []
    quat_list = []
    omega_list = []
    energy_history_list = []
    cumulative_energy_joules = np.zeros(6)
    instant_power_list = []
    dominant_dir_list = []

    t_grid = tofs.to_value(u.s)

    ## Initial R (1)
    R = np.eye(3)
    # Converts Body -> Inertial

    # Propagation Loop
    for i in range(1, len(t_grid)):
        ## Time Setup
        # Current time
        t = t_grid[i]

        # Current position and velocity vector:
        r = rr[i].to(u.km).value
        v = vv[i].to(u.km / u.s).value

        # Current state (translational)
        full_current_state = np.concatenate((r, v))

        # Current state (rotational)
        spin_current_state = np.concatenate((quat, omega))

        gcrs_coords = full_current_state[0:3]

        # Timestep (s)
        dt = t_grid[i] - t_grid[i - 1]

        ## Torque Integration
        # External Torques

        # Finding drag acceleration using consistent timesteps
        r_mag_prop = np.linalg.norm(r)
        alt_km_prop = r_mag_prop - Earth.R.to(u.km).value

        # Get rho at this altitude
        log_rho_prop = get_log_rho(alt_km_prop)
        rho_prop = np.exp(log_rho_prop)

        omega_Earth = np.array([0, 0, omega_E])

        v_atm_prop = np.cross(omega_Earth, r)

        v_rel_prop = v - v_atm_prop

        a_drag_vec = (
                -0.5
                * C_D
                * (A * 1e-6 / m)  # We multiply by 1e-06 to output in km/s/s (for readability purposes, because we're just going to convert to SI units in the next step)
                * rho_prop
                * np.linalg.norm(v_rel_prop)
                * v_rel_prop
        )

        F_drag = m * a_drag_vec * 1000  # To change from km/s^2 to m/s^2

        F_body = R.T @ F_drag

        r_cp_cm = np.array([0.03, 0, 0])
        Tau_drag = np.cross(r_cp_cm, F_body)

        # Magnetic Torque (barbeque roll)
        """
        Tau_magnet, alignment_angle, u_mag_vector= magnetic_torque(t, full_current_state, R)

        Tau_damp = damper_torque(spin_current_state, u_mag_vector, R)
        """
        Tau_magnet, alignment_angle, u_mag_vector = magnetic_torque(t, full_current_state, R)

        Tau_damp = damper_torque(spin_current_state)

        # Tau_vec = (Tau_drag)
        Tau_vec = Tau_drag + Tau_magnet + Tau_damp

        '''
        # Debugging
        print("Tau_drag =", Tau_drag)
        print("Tau_magnet =", Tau_magnet)
        print("I_tensor (det) =", np.linalg.det(I_tensor))
        print("I_tensor=", I_tensor)
        '''

        ## Spin Integration
        # Instead of using euler_eq directly, we'll be using an RK4 Method for the rotational integration:
        spin_current_state = rk4_spin(t, spin_current_state, dt, I_tensor, Tau_vec)

        # Splitting the spin state into quaternion and omega
        quat = spin_current_state[0:4]
        omega = spin_current_state[4:7]


        ## Debugging (Can remove if produces no output)
        if i % 500 == 0:
            print("Drag tau:", np.linalg.norm(Tau_drag), #"Magnet tau:", np.linalg.norm(Tau_magnet),
             "Tau:", np.linalg.norm(Tau_vec))
            print("Drag Tau =", Tau_drag, "Tau Vector =", Tau_vec)

        ## Debugging (Can remove if produces no output)
        if not np.all(np.isfinite(omega)):
            print("Bad Omega lol")
            print('t=',t)
            print('dt=',dt)
            print('omega=',omega)
            break

        ## Debugging (Can comment out or remove if desired)
        if i % 500 == 0:
            print(
                f"t={t:.0f}",
                f"|omega|={np.linalg.norm(omega):.4f}; omega={omega}"
            )

        # Debugging (Can remove if desired)
        if i % 500 == 0:
            print("Quaternion:", quat, "mag =", np.linalg.norm(quat))

        ## Debugging
        if not np.all(np.isfinite(quat)):
            print("Bad Quaternion lol")
            print('t=',t)
            print('quat=', quat)
            break

        """
        # Printing Alignment angle:
        if i % 500 == 0:
            print(
                f"Alignment Angle at t = {t:.0f} is {alignment_angle:.2f} degrees"
            )
        """

        # spin values required for both derivatives() and calculating drag_force in the body frame
        Earth_cosine, Sun_cosine, sun_state, R = spin_sun_earth(t, quat, gcrs_coords)

        ## Temperature Integration
        dT_faces, dTi_dt, P_faces = derivatives(t, T_faces, T_internal, full_current_state, Earth_cosine, Sun_cosine, sun_state)

        T_faces += dT_faces * dt
        T_internal += dTi_dt * dt

        if not np.all(np.isfinite(T_faces)):
            print("Bad Temperature lol")
            print(t)
            print(T_faces)
            break

        # Power Integration
        # Power (Watts = Joules/sec) * dt (seconds) = Energy generated this step (Joules)
        # Log instant power generation
        instant_power_list.append(P_faces.copy())
        energy_step_joules = P_faces * dt

        # Add it to our running total for each face
        cumulative_energy_joules += energy_step_joules

        # Optional: Convert Joules to Watt-hours (Wh)
        # (1 Watt-hour = 3,600 Joules)
        cumulative_energy_wh = cumulative_energy_joules / 3600.0

        #Dominant rotation direction at any given time -- time series list
        om = omega.copy()
        wx, wy, wz = om

        #Key: Dominant direction 1 is wx is dominant, 2 for wy, and 3 for wz

        if wx > wy and wx > wz:
            dominant_direction = 1
        elif wy>wx and wy>wz:
            dominant_direction = 2
        elif wz>wy and wy>wx:
            dominant_direction = 3


        ## Update Lists
        ''' So apparently there's a bug ("") in numpy, where if you were to append a numpy list, it might change the previous quantity
        This is because numpy lists are mutable, and refer to the id of the object, not the value. Just to be on the safe side, we'll be
        using .copy(). This doesn't change any values; it's just a precaution.'''
        T_ext_list.append(T_faces.copy())
        T_int_list.append(float(T_internal))
        quat_list.append(quat.copy())
        omega_list.append(omega.copy())
        energy_history_list.append(cumulative_energy_wh.copy())
        dominant_dir_list.append(dominant_direction)

    return np.array(T_ext_list), np.array(T_int_list), omega_list, np.array(energy_history_list), np.array(instant_power_list), np.array(dominant_dir_list), np.array(quat_list)


### Running the Thermal Model (AFTER integrating f function)
T_ext, T_int, omega, energy_history_list, instant_pwr_history, dominant_direction_history, quat_history = propagation(tofs, rr, vv)

### Maximum and Minimum Temperatures
# External panels
Te_max = np.max(T_ext, axis=0)
Te_min = np.min(T_ext, axis=0)

# Internal node
Ti_max = np.max(T_int)
Ti_min = np.min(T_int)

print("\nExternal Panel Temperature Extremes:")
for i in range(6):
    print(f"Panel {i + 1}: min = {Te_min[i]:.3f} K "
          f"(or {(Te_min[i] - 273.15):.3f} °C); "
          f"max = {Te_max[i]:.3f} K "
          f"(or {(Te_max[i] - 273.15):.3f} °C)."
          )

print("\nInternal Temperature Extremes:")
print(f"Internal: min = {Ti_min:.3f} "
      f"(or {(Ti_min - 273.15):.3f} °C) "
      f"K; max = {Ti_max:.3f} K "
      f"(or {(Ti_max - 273.15):.3f} °C).\n"
      )


### Thermal Expansion function
# Birch is anisotropic, so it has different values across and along the grain
def thermal_expansion(Lx_panel, Ly_panel, Lz_panel, Te_max, Te_min):
    ## Thermal Expansion variables (K^-1). Alpha here is NOT angular acceleration
    alpha_across = 30e-6
    alpha_along = 3e-6

    # Lx and Ly of the panels are area (10 cm each)
    # Lz of the panels is the thickness (6 mm)

    # Variable for Temperature during building (when the material is the "least" stressed as a reference)
    T_build = 297.15  # K (or 24 °C)

    # Array for each of the 6 panels values
    # Hot cases
    dLx_list_hot = np.zeros(6)
    dLy_list_hot = np.zeros(6)
    dLz_list_hot = np.zeros(6)

    # Cold cases
    dLx_list_cold = np.zeros(6)
    dLy_list_cold = np.zeros(6)
    dLz_list_cold = np.zeros(6)

    print(f"\nThermal Expansion (relative to {T_build - 273.15} °C build temperature)")

    # Loop for each of the panels (because they each receive a different temperature)
    for i in range(6):
        dT_hot = Te_max[i] - T_build
        dT_cold = Te_min[i] - T_build

        # Calculating each of the panels
        dLx_hot = alpha_across * Lx_panel * dT_hot
        dLy_hot = alpha_across * Ly_panel * dT_hot
        dLz_hot = alpha_along * Lz_panel * dT_hot

        dLx_cold = alpha_across * Lx_panel * dT_cold
        dLy_cold = alpha_across * Ly_panel * dT_cold
        dLz_cold = alpha_along * Lz_panel * dT_cold

        # Adding the values to the list
        dLx_list_hot[i] = dLx_hot
        dLy_list_hot[i] = dLy_hot
        dLz_list_hot[i] = dLz_hot

        dLx_list_cold[i] = dLx_cold
        dLy_list_cold[i] = dLy_cold
        dLz_list_cold[i] = dLz_cold

        print(f"Panel {i + 1}")
        print(f"  Hot Case: ")
        print(f"    ΔT = {dT_hot:.3f} K")
        print(f"    ΔLx = {dLx_hot * 1e3:.6f} mm")
        print(f"    ΔLy = {dLy_hot * 1e3:.6f} mm")
        print(f"    ΔLz = {dLz_hot * 1e3:.6f} mm")

        print(f"\n  Cold Case: ")
        print(f"    ΔT = {dT_cold:.3f} K")
        print(f"    ΔLx = {dLx_cold * 1e3:.6f} mm")
        print(f"    ΔLy = {dLy_cold * 1e3:.6f} mm")
        print(f"    ΔLz = {dLz_cold * 1e3:.6f} mm")

        print("\nTotal Swing:")
        print(f"    ΔLx = {(dLx_hot - dLx_cold) * 1e3:.6f} mm")
        print(f"    ΔLy = {(dLy_hot - dLy_cold) * 1e3:.6f} mm")
        print(f"    ΔLz = {(dLz_hot - dLz_cold) * 1e3:.6f} mm")

    # Differential Analysis
    print("\nDifferential Expansion Analysis:")
    print("\nHOT CASE:")
    print(f"\nDifference in X-panels (panels 1 and 2 (or 0 and 1 in the model) = face2 - face1):")
    print(f"    x-expansion: {(dLx_list_hot[1] - dLx_list_hot[0]) * 1e3:.6f} mm")
    print(f"    y-expansion: {(dLy_list_hot[1] - dLy_list_hot[0]) * 1e3:.6f} mm")
    print(f"    z-expansion: {(dLz_list_hot[1] - dLz_list_hot[0]) * 1e3:.6f} mm")
    print(f"\nDifference in Y-panels (panels 3 and 4 (or 2 and 3 in the model) = face4 - face3):")
    print(f"    x-expansion: {(dLx_list_hot[3] - dLx_list_hot[2]) * 1e3:.6f} mm")
    print(f"    y-expansion: {(dLy_list_hot[3] - dLy_list_hot[2]) * 1e3:.6f} mm")
    print(f"    z-expansion: {(dLz_list_hot[3] - dLz_list_hot[2]) * 1e3:.6f} mm")
    print(f"\nDifference in Z-panels (panels 5 and 6 (or 4 and 5 in the model) = face6 - face5):")
    print(f"    x-expansion: {(dLx_list_hot[5] - dLx_list_hot[4]) * 1e3:.6f} mm")
    print(f"    y-expansion: {(dLy_list_hot[5] - dLy_list_hot[4]) * 1e3:.6f} mm")
    print(f"    z-expansion: {(dLz_list_hot[5] - dLz_list_hot[4]) * 1e3:.6f} mm")

    print("\n\nCOLD CASE:")
    print(f"\nDifference in X-panels (panels 1 and 2 (or 0 and 1 in the model) = face2 - face1):")
    print(f"    x-expansion: {(dLx_list_cold[1] - dLx_list_cold[0]) * 1e3:.6f} mm")
    print(f"    y-expansion: {(dLy_list_cold[1] - dLy_list_cold[0]) * 1e3:.6f} mm")
    print(f"    z-expansion: {(dLz_list_cold[1] - dLz_list_cold[0]) * 1e3:.6f} mm")
    print(f"\nDifference in Y-panels (panels 3 and 4 (or 2 and 3 in the model) = face4 - face3):")
    print(f"    x-expansion: {(dLx_list_cold[3] - dLx_list_cold[2]) * 1e3:.6f} mm")
    print(f"    y-expansion: {(dLy_list_cold[3] - dLy_list_cold[2]) * 1e3:.6f} mm")
    print(f"    z-expansion: {(dLz_list_cold[3] - dLz_list_cold[2]) * 1e3:.6f} mm")
    print(f"\nDifference in Z-panels (panels 5 and 6 (or 4 and 5 in the model) = face6 - face5):")
    print(f"    x-expansion: {(dLx_list_cold[5] - dLx_list_cold[4]) * 1e3:.6f} mm")
    print(f"    y-expansion: {(dLy_list_cold[5] - dLy_list_cold[4]) * 1e3:.6f} mm")
    print(f"    z-expansion: {(dLz_list_cold[5] - dLz_list_cold[4]) * 1e3:.6f} mm")

    return dLx_list_hot, dLy_list_hot, dLz_list_hot, dLx_list_cold, dLy_list_cold, dLz_list_cold


## Running the Thermal Expansion loop
dLx_list_hot, dLy_list_hot, dLz_list_hot, dLx_list_cold, dLy_list_cold, dLz_list_cold = thermal_expansion(Lx_panel,
                                                                                                          Ly_panel,
                                                                                                          Lz_panel,
                                                                                                          Te_max,
                                                                                                          Te_min)

# 3D Render
plotter = OrbitPlotter(backend=Plotly3D())
plotter.set_attractor(Earth)
plotter.plot(orbit, label="WS2 Flight Path", color="#ff5733")
plotter.show()
# Altitude Plot
plt.ylabel("h(t)")
plt.xlabel("t, days")
plt.plot(tofs[: len(rr)].value, np.linalg.norm(rr, axis=1) - Earth.R)
plt.show()

# Printing Maximum and minimum values
altitude = np.linalg.norm(
    rr.to(u.km).value,
    axis=1
) - Earth.R.to(u.km).value

print("Max Altitude:", np.max(altitude), "And Min Altitude:", np.min(altitude))

# Power generation per face plot
face_names = ["+X Face", "-X Face", "+Y Face", "-Y Face", "+Z Face", "-Z Face"]
fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axes = axes.flatten()  # Flattens the 3x2 grid into a simple 1D array of 6 axes

time_grid = tofs.to_value(u.s)
fig1, axes1 = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axes1 = axes1.flatten()
time_axis = time_grid[1:]

# Loop through each face and plot its energy history
for i in range(6):
    # energy_history[:, i] slices out just the column for face 'i' across all timesteps
    axes[i].plot(time_axis, energy_history_list[:, i], color='tab:blue', linewidth=1.5)

    axes[i].set_title(f"Cumulative Energy: {face_names[i]}")
    axes[i].set_ylabel("Energy (Wh)")
    axes[i].grid(True)

# Add x-axis labels to the bottom plots
axes[4].set_xlabel("Time (s)")
axes[5].set_xlabel("Time (s)")

plt.tight_layout()
plt.show()

print("Power generated by", face_names[0], ":", f"{energy_history_list[-1, 0]:.4f} Wh")
print("Power generated by", face_names[1], ":", f"{energy_history_list[-1, 1]:.4f} Wh")
print("Power generated by", face_names[2], ":", f"{energy_history_list[-1, 2]:.4f} Wh")
print("Power generated by", face_names[3], ":", f"{energy_history_list[-1, 3]:.4f} Wh")
print("Power generated by", face_names[4], ":", f"{energy_history_list[-1, 4]:.4f} Wh")
print("Power generated by", face_names[5], ":", f"{energy_history_list[-1, 5]:.4f} Wh")

print("Power generated by", face_names[0], ":", f"{energy_history_list[-1, 0]:.4f} Wh")
print("Power generated by", face_names[1], ":", f"{energy_history_list[-1, 1]:.4f} Wh")
print("Power generated by", face_names[2], ":", f"{energy_history_list[-1, 2]:.4f} Wh")
print("Power generated by", face_names[3], ":", f"{energy_history_list[-1, 3]:.4f} Wh")
print("Power generated by", face_names[4], ":", f"{energy_history_list[-1, 4]:.4f} Wh")
print("Power generated by", face_names[5], ":", f"{energy_history_list[-1, 5]:.4f} Wh")

# Time axis (skipping the first history time since the loop starts at index 1)
time_grid = tofs.to_value(u.s)
time_axis = time_grid[1:]

# --- 2. SECOND PLOT: Instantaneous Power ---
fig2, axes2 = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axes2 = axes2.flatten()

for i in range(6):
    axes2[i].plot(time_axis, instant_pwr_history[:, i], color='tab:orange', linewidth=1.5)
    axes2[i].set_title(f"Instantaneous Power: {face_names[i]}")
    axes2[i].set_ylabel("Power (W)")
    axes2[i].grid(True)

axes2[4].set_xlabel("Time (s)")
axes2[5].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()

# --- 3. THIRD PLOT: Dominant Direction. Remember the key from earlier, same applies.
time_grid = tofs.to_value(u.s)
time_axis = time_grid[1:]

fig3, axes3 = plt.subplots(figsize=(12, 10))
axes3.plot(time_axis, dominant_direction_history, color='tab:blue', linewidth=1.5)
axes3.set_title("Dominant Direction. 1=wx, 2=wy, 3=wz")
axes3.set_xlabel("Time (s)")
axes3.set_ylabel("Numerical Indicator for Dominant Direction")
axes3.grid(True)
axes3.legend()
plt.show()

#
# # -- 4. Fourth item: 3d vector visualization thingy
# time_grid = tofs.to_value(u.s)
# time_axis = time_grid[1:]
#
# fig4, axes4 = plt.subplots(subplot_kw=dict(projection="3d"))
# axes4: Axes3D = axes4
#
# axes4.set_xlim(-2, 2)
# axes4.set_ylim(-2, 2)
# axes4.set_zlim(-2, 2)
#
# X_axis = [1,0,0]
# Y_axis = [0,1,0]
# Z_axis = [0,0,1]
#
# U_X_in = [1,0,0]
# U_Y_in = [0,1,0]
# U_Z_in = [0,0,1]
#
# quiver = axes4.quiver(X_axis, Y_axis, Z_axis, U_X_in, U_Y_in, U_Z_in)
#
# def get_arrows(r_anim):
#     X_ax =
#     Y_ax =
#     Z_ax =
#
#
#     return X_ax, Y_ax, Z_ax
#
#
# def update(i):
#     quiver.remove()
#     r_anim = R.from_quat(quat_history[i])
#
#
#
#
# axes4.set_xlabel("x")
# axes4.set_ylabel("y")
# axes4.set_zlabel("z")
#
# """
# Construction Reference for how many frames we need to keep everything synced
# time_grid = tofs.to_value(u.s)
# time_axis = time_grid[1:]
#
# The propogation loop didn't run for the 'initial' state, so quat_history should have one less item that tofs. Hopefully
# this allows for time_axis and quat_history to be called together with the same 'i' indicator, since FuncAnimation
# doesn't start at frame 0 either, I hope."""
#
# anim = FuncAnimation(fig4, update, frames=time_axis, interval=50, repeat=False, blit=True)
#
