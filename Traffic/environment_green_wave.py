"""Grid/green wave example."""
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter

# time horizon of a single rollout
HORIZON = 1000



def gen_edges(col_num, row_num):
    """Generate the names of the outer edges in the grid network.

    Parameters
    ----------
    col_num : int
        number of columns in the grid
    row_num : int
        number of rows in the grid

    Returns
    -------
    list of str
        names of all the outer edges
    """
    edges = []
    for i in range(col_num):
        edges += ['left' + str(row_num) + '_' + str(i)]
        edges += ['right' + '0' + '_' + str(i)]

    # build the left and then the right edges
    for i in range(row_num):
        edges += ['bot' + str(i) + '_' + '0']
        edges += ['top' + str(i) + '_' + str(col_num)]

    return edges


def get_flow_params(col_num, row_num, additional_net_params):
    """Define the network and initial params in the presence of inflows.

    Parameters
    ----------
    col_num : int
        number of columns in the grid
    row_num : int
        number of rows in the grid
    additional_net_params : dict
        network-specific parameters that are unique to the grid

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the scenario
    """
    initial = InitialConfig(
        spacing='custom', lanes_distribution=float('inf'), shuffle=True)

    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)
    for i in range(len(outer_edges)):
        inflow.add(
            veh_type='idm',
            edge=outer_edges[i],
            probability=0.25,
            departLane='free',
            departSpeed=10)

    net = NetParams(
        inflows=inflow,
        additional_params=additional_net_params)

    return initial, net


def get_non_flow_params(enter_speed, add_net_params):
    """Define the network and initial params in the absence of inflows.

    Note that when a vehicle leaves a network in this case, it is immediately
    returns to the start of the row/column it was traversing, and in the same
    direction as it was before.

    Parameters
    ----------
    enter_speed : float
        initial speed of vehicles as they enter the network.
    add_net_params: dict
        additional network-specific parameters (unique to the grid)

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the scenario
    """
    additional_init_params = {'enter_speed': enter_speed}
    initial = InitialConfig(
        spacing='custom', additional_params=additional_init_params)
    net = NetParams(additional_params=add_net_params)

    return initial, net


V_ENTER = 15
INNER_LENGTH = 100
LONG_LENGTH = 100
SHORT_LENGTH = 600
N_ROWS = 2
N_COLUMNS = 3
NUM_CARS_LEFT = 50
NUM_CARS_RIGHT = 50
NUM_CARS_TOP = 5
NUM_CARS_BOT = 5
tot_cars = (NUM_CARS_LEFT + NUM_CARS_RIGHT) * N_COLUMNS \
           + (NUM_CARS_BOT + NUM_CARS_TOP) * N_ROWS

grid_array = {
    "short_length": SHORT_LENGTH,
    "inner_length": INNER_LENGTH,
    "long_length": LONG_LENGTH,
    "row_num": N_ROWS,
    "col_num": N_COLUMNS,
    "cars_left": NUM_CARS_LEFT,
    "cars_right": NUM_CARS_RIGHT,
    "cars_top": NUM_CARS_TOP,
    "cars_bot": NUM_CARS_BOT
}

additional_env_params = {
        'target_velocity': 35,
        'switch_time': 3.0,
        'num_observed': 2,
        'discrete': False,
        'max_accel': 3,
        # maximum deceleration for autonomous vehicles, in m/s^2
        'max_decel': 3,
        # specifies whether vehicles are to be sorted by position during a
        # simulation step. If set to True, the environment parameter
        # self.sorted_ids will return a list of all vehicles sorted in accordance
        # with the environment
        'sort_vehicles': False,
        'tl_type': 'controlled'
    }

additional_net_params = {
    'speed_limit': 35,
    'grid_array': grid_array,
    'horizontal_lanes': 1,
    'vertical_lanes': 1
}

vehicles = VehicleParams()
vehicles.add(
    veh_id='idm',
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
        minGap=2.5,
        decel=7.5,  # avoid collisions at emergency stops
        max_speed=V_ENTER,
        speed_mode="all_checks",
    ),
    routing_controller=(GridRouter, {}),
    num_vehicles=tot_cars)

flow_params = dict(
    # name of the experiment
    exp_tag='green_wave',

    # name of the flow environment the experiment is running on
    env_name='PO_TrafficLightGridEnv',

    # name of the scenario class the experiment is running on
    scenario='SimpleGridScenario',

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.2,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params=additional_env_params,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component). This is
    # filled in by the setup_exps method below.
    net=None,

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig). This is filled in by the
    # setup_exps method below.
    initial=None,
)


def setup_exps(use_inflows=False):
    """Return the relevant components of an RLlib experiment.

    Parameters
    ----------
    use_inflows : bool, optional
        set to True if you would like to run the experiment with inflows of
        vehicles from the edges, and False otherwise

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    # collect the initialization and network-specific parameters based on the
    # choice to use inflows or not
    if use_inflows:
        initial_config, net_params = get_flow_params(
            col_num=N_COLUMNS,
            row_num=N_ROWS,
            additional_net_params=additional_net_params)
    else:
        initial_config, net_params = get_non_flow_params(
            enter_speed=V_ENTER,
            add_net_params=additional_net_params)

    # add the new parameters to flow_params
    flow_params['initial'] = initial_config
    flow_params['net'] = net_params
    create_env, gym_name = make_create_env(params=flow_params, version=0)
    TheEnv=create_env()
    return TheEnv


def TL_env():
    envTL = setup_exps(use_inflows=False)
    return envTL
