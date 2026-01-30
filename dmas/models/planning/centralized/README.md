# Centralized Planning Models
<!-- 3DCHESS Concept of Operations and Mission Outline -->
In the 3DCHESS project, we are envisioning scenarios where we have satellites actively trying to detect and re-observe geophysical events that are located in previously defined points of interest. Satellites may be part of different missions with different priorities of observation for said events depending on their own mission priorities and objectives. To help improve the response and coordination of said satellite missions, the effects of increased satellite autonomy on mission performance is of great interest.

Satellites (or agents) are therefore tasked with detecting previously unknown events of interest while also re-observing previously detected events. Due to the predictable nature of satellite coverage metrics and of orbital flight maneuvers, observation tasks can be scheduled in an open-loop manner where an observation and attitude maneuver sequence can be planned in advance and then executed with very predictable results. These sequences could be planned autonomously by either a centralized automated planner or an onboard planner. 

In this section we present different approaches to how centralized planning for satellite operations planning and scheduling could be implemented in this complex mission context. Centralized operations would allow for a potentially higher fidelity plan to be generated as computational resources can be concentrated to a single agent, but could be limited by the latencyt in access times between agents which could lead to outdated plans being executed. 

## Mixed-Integer Linear Programming Approach

An comparative overview of the propsed MILP models is as follows: 

| Model Name  | Time Assignment          | Reward Function                       | Reobs / Revisit Handling        | Notes                                      |
|-------------|--------------------------|---------------------------------------|---------------------------------|--------------------------------------------|
| Static      | At start of access window | Static, precomputed per task          | None                            | Simplest, fastest; ignores dynamics        |
| Linear      | Any time during access window       | Linear variation across access window | None                            | Captures time-dependence, no reobs         |
| Reobs       | Any time during access window       | Depends on # of reobservations + time of observation       | Reobs only                      | Tracks constellation-wide redundancy       |
| Revisit     | Any time during access window       | Depends on reobs + revisit intervals + time of observation  | Reobs + Revisit timing          | Richest model; most realistic, most costly |


### Static Reward with Fixed Observation Time Model

### Linear Reward with Varying Observation Time Model

### Number of Reobservation Dependent Model 

### Reobservation Time and Number Dependent Model