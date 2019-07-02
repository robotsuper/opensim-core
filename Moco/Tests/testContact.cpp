/* -------------------------------------------------------------------------- *
 * OpenSim Moco: testContact.cpp                                              *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2017 Stanford University and the Authors                     *
 *                                                                            *
 * Author(s): Christopher Dembia                                              *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may    *
 * not use this file except in compliance with the License. You may obtain a  *
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0          *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 * -------------------------------------------------------------------------- */

// TODO add 3D tests (contact models are currently only 2D).

#define CATCH_CONFIG_MAIN
#include "Testing.h"

#include <Moco/osimMoco.h>
#include <OpenSim/Simulation/Manager/Manager.h>
#include <Moco/Components/SmoothSphereHalfSpaceForce.h>

const double FRICTION_COEFFICIENT = 0.7;

using namespace OpenSim;
using SimTK::Vec3;

using CreateContactFunction = std::function<StationPlaneContactForce*(void)>;
Model create2DPointMassModel(CreateContactFunction createContact) {
    Model model;
    model.setName("point_mass");
    auto* intermed = new Body("intermed", 0, Vec3(0), SimTK::Inertia(0));
    model.addComponent(intermed);
    auto* body = new Body("body", 50.0, Vec3(0), SimTK::Inertia(1));
    model.addComponent(body);

    // Allows translation along x.
    auto* jointX = new SliderJoint();
    jointX->setName("tx");
    jointX->connectSocket_parent_frame(model.getGround());
    jointX->connectSocket_child_frame(*intermed);
    auto& coordX = jointX->updCoordinate(SliderJoint::Coord::TranslationX);
    coordX.setName("tx");
    model.addComponent(jointX);

    // The joint's x axis must point in the global "+y" direction.
    auto* jointY = new SliderJoint("ty",
            *intermed, Vec3(0), Vec3(0, 0, 0.5 * SimTK::Pi),
            *body, Vec3(0), Vec3(0, 0, 0.5 * SimTK::Pi));
    auto& coordY = jointY->updCoordinate(SliderJoint::Coord::TranslationX);
    coordY.setName("ty");
    model.addComponent(jointY);

    auto* station = new Station();
    station->setName("contact_point");
    station->connectSocket_parent_frame(*body);
    model.addComponent(station);

    auto* force = createContact();
    force->setName("contact");
    model.addComponent(force);
    force->connectSocket_station(*station);

    return model;
}

// Test that, with dissipation, the contact force settles to the weight of the
// system. This is the same type of test done in OpenSim's testForces for
// HuntCrossleyForce. The test is performed with both time stepping and direct
// collocation.
SimTK::Real testNormalForce(CreateContactFunction createContact) {
    Model model(create2DPointMassModel(createContact));
    SimTK::Real weight;
    {
        SimTK::State state = model.initSystem();
        weight = model.getTotalMass(state) * (-model.getGravity()[1]);
    }

    const SimTK::Real y0 = 0.5;
    const SimTK::Real finalTime = 2.0;

    // Time stepping.
    // --------------
    SimTK::Real finalHeightTimeStepping;
    {
        SimTK::State state = model.initSystem();
        model.setStateVariableValue(state, "ty/ty/value", y0);
        Manager manager(model);
        manager.setIntegratorAccuracy(1e-6);
        manager.initialize(state);
        state = manager.integrate(finalTime);

        // visualize(model, manager.getStateStorage());

        auto& contact = model.getComponent<StationPlaneContactForce>("contact");
        model.realizeVelocity(state);
        const Vec3 contactForce = contact.calcContactForceOnStation(state);
        // The horizontal force is not quite zero, maybe from a buildup of
        // numerical error (tightening the accuracy reduces this force).
        CHECK(contactForce[0] == Approx(0).margin(0.01));
        CHECK(contactForce[1] == Approx(weight).epsilon(0.01));
        // The system is planar, so there is no force in the z direction.
        CHECK(contactForce[2] == 0);

        finalHeightTimeStepping =
                model.getStateVariableValue(state, "ty/ty/value");
    }

    // Direct collocation.
    // -------------------
    // This is a simulation (initial value problem), not a trajectory
    // optimization.
    SimTK::Real finalHeightDircol;
    {
        MocoStudy moco;
        MocoProblem& mp = moco.updProblem();
        mp.setModelCopy(model);
        mp.setTimeBounds(0, finalTime);
        mp.setStateInfo("/tx/tx/value", {-1, 1}, 0);
        mp.setStateInfo("/ty/ty/value", {-0.5, 1}, y0);
        mp.setStateInfo("/tx/tx/speed", {-10, 10}, 0);
        mp.setStateInfo("/ty/ty/speed", {-10, 10}, 0);

        auto& ms = moco.initTropterSolver();
        ms.set_num_mesh_points(50);
        // TODO: Hermite-Simpson has trouble converging
        ms.set_transcription_scheme("trapezoidal");

        MocoSolution solution = moco.solve();
        solution.write("testContact_solution_testNormalForce.sto");
        // moco.visualize(solution);

        auto statesTraj = solution.exportToStatesTrajectory(mp);
        const auto& finalState = statesTraj.back();
        model.realizeVelocity(finalState);
        auto& contact = model.getComponent<StationPlaneContactForce>("contact");
        const Vec3 contactForce = contact.calcContactForceOnStation(finalState);
        // For some reason, direct collocation doesn't produce the same
        // numerical issues with the x component of the force as seen above.
        CHECK(contactForce[0] == Approx(0).margin(1e-15));
        CHECK(contactForce[1] == Approx(weight).epsilon(0.01));
        CHECK(contactForce[2] == 0);

        finalHeightDircol =
                model.getStateVariableValue(finalState, "ty/ty/value");
    }

    CHECK(finalHeightTimeStepping == Approx(finalHeightDircol).margin(1e-5));

    return finalHeightTimeStepping;
}

// Test the friction component of the contact force by ensuring that the point
// mass travels the expected horizontal distance if it starts in the ground.
// To make the friction force roughly constant, we want the equilibrium height
// of the mass (from testNormalForce()).
void testFrictionForce(CreateContactFunction createContact,
        const SimTK::Real& equilibriumHeight) {
    Model model(create2DPointMassModel(createContact));
    {
        SimTK::State state = model.initSystem();
    }

    const SimTK::Real y0 = equilibriumHeight;
    const SimTK::Real finalTime = 0.5;
    const SimTK::Real vx0 = 2.5;

    const SimTK::Real g = -model.getGravity()[1];

    // Expected final x position.
    // --------------------------
    // m * vxdot = F = - mu * m * g
    // vx(t) = -mu * g * t + vx0
    // x(t) = -0.5 * mu * g * t^2 + vx0 * t
    // The time at which the point reaches rest:
    // t_rest = vx0 / (mu * g)
    // Final position: x(t_rest)
    const double& mu = FRICTION_COEFFICIENT;
    const SimTK::Real restTime = vx0 / (mu * g);
    CHECK(restTime < finalTime);
    const SimTK::Real expectedFinalX =
            -0.5 * mu * g * pow(restTime, 2) + vx0 * restTime;

    // Time stepping.
    // --------------
    {
        SimTK::State state = model.initSystem();
        model.setStateVariableValue(state, "ty/ty/value", y0);
        model.setStateVariableValue(state, "tx/tx/speed", vx0);
        Manager manager(model, state);
        state = manager.integrate(finalTime);

        // visualize(model, manager.getStateStorage());

        const SimTK::Real finalTX =
                model.getStateVariableValue(state, "tx/tx/value");
        CHECK(finalTX == Approx(expectedFinalX).margin(0.005));

        // The system should be at rest.
        OpenSim_CHECK_MATRIX_ABSTOL(state.getU(),
                SimTK::Vector(state.getNU(), 0.0), 1e-3);
    }

    // Direct collocation.
    // -------------------
    // This is a simulation (initial value problem), not a trajectory
    // optimization.
    {
        MocoStudy moco;
        MocoProblem& mp = moco.updProblem();
        mp.setModelCopy(model);
        mp.setTimeBounds(0, finalTime);
        mp.setStateInfo("/tx/tx/value", {-1, 1}, 0);
        mp.setStateInfo("/ty/ty/value", {-0.5, 1}, y0);
        mp.setStateInfo("/tx/tx/speed", {-10, 10}, vx0);
        mp.setStateInfo("/ty/ty/speed", {-10, 10}, 0);

        auto& ms = moco.initTropterSolver();
        ms.set_num_mesh_points(25);
        // TODO: Hermite-Simpson has trouble converging
        ms.set_transcription_scheme("trapezoidal");

        MocoSolution solution = moco.solve();
        solution.write("testContact_testFrictionForce_solution.sto");
        // moco.visualize(solution);

        auto statesTraj = solution.exportToStatesTrajectory(mp);
        const auto& finalState = statesTraj.back();
        const SimTK::Real finalTX =
                model.getStateVariableValue(finalState, "tx/tx/value");

        CHECK(finalTX == Approx(expectedFinalX).margin(0.005));

        // The system should be at rest.
        OpenSim_CHECK_MATRIX_ABSTOL(finalState.getU(),
                SimTK::Vector(finalState.getNU(), 0.0), 1e-3);
    }
}

void testStationPlaneContactForce(CreateContactFunction createContact) {
    const SimTK::Real equilibriumHeight = testNormalForce(createContact);
    testFrictionForce(createContact, equilibriumHeight);
}

// Test our wrapping of SmoothSphereHalfSpaceForce in Moco
// Create a model with SmoothSphereHalfSpaceForce
Model createBallHalfSpaceModel() {
    // Setup model.
    Model model;
    model.setName("BouncingBall_SmoothSphereHalfSpaceForce");
    auto* ball = new Body("ball", 1, Vec3(0), SimTK::Inertia(1));
    model.addComponent(ball);
    auto* groundBall = new PlanarJoint("groundBall",model.getGround(),
    Vec3(0), Vec3(0), *ball, Vec3(0), Vec3(0));
    model.addComponent(groundBall);
    // Add display geometry.
    Sphere bodyGeometry(0.5);
    bodyGeometry.setColor(SimTK::Gray);
    auto* ballCenter = new PhysicalOffsetFrame(
    "ballCenter", *ball, Transform(Vec3(0)));
    ball->addComponent(ballCenter);
    ballCenter->attachGeometry(bodyGeometry.clone());
    // Setup contact.
    double radius = 0.5;
    double stiffness = 10000;
    double dissipation = 0.75;
    double staticFriction = FRICTION_COEFFICIENT;
    double dynamicFriction = FRICTION_COEFFICIENT;
    double viscousFriction = FRICTION_COEFFICIENT;
    double transitionVelocity = 0.1;
    double cf = 1e-5;
    double bd = 300;
    double bv = 50;
    Vec3 sphereLocation(0);
    SimTK::Transform halfSpaceFrame(Rotation(0, SimTK::ZAxis), Vec3(0));
    auto* contactBallHalfSpace = new SmoothSphereHalfSpaceForce(
        "contactBallHalfSpace",*ball,sphereLocation,radius,model.getGround(),
        halfSpaceFrame,stiffness,dissipation,staticFriction,dynamicFriction,
        viscousFriction,transitionVelocity,cf,bd,bv);
    contactBallHalfSpace->setName("contactBallHalfSpace");
    model.addComponent(contactBallHalfSpace);
    contactBallHalfSpace->connectSocket_body_contact_sphere(*ball);
    contactBallHalfSpace->connectSocket_body_contact_half_space(
        model.getGround());

    return model;
}

// Simple simulation of bouncing ball with dissipation should generate contact
// forces that settle to ball weight. This is the same type of test done in
// OpenSim's testForces for HuntCrossleyForce. The test is performed with both
// time stepping and direct collocation.
SimTK::Real testSmoothSphereHalfSpaceForce_NormalForce()
{
    Model model(createBallHalfSpaceModel());

    SimTK::Real weight;
    {
        SimTK::State state = model.initSystem();
        weight = model.getTotalMass(state) * (-model.getGravity()[1]);
    }

    const SimTK::Real y0 = 0.5;
    const SimTK::Real finalTime = 2.0;

    // Time stepping.
    // --------------
    SimTK::Real finalHeightTimeStepping;
    {
        SimTK::State state = model.initSystem();
        model.setStateVariableValue(state,
            "groundBall/groundBall_coord_2/value", y0);
        Manager manager(model);
        manager.setIntegratorAccuracy(1e-6);
        manager.initialize(state);
        state = manager.integrate(finalTime);

        model.realizeVelocity(state);

        auto& contactBallHalfSpace =
            model.getComponent<SmoothSphereHalfSpaceForce>(
                "contactBallHalfSpace");

        Array<double> contactForces =
            contactBallHalfSpace.getRecordValues(state);
        CHECK(contactForces[0] == Approx(0.0).margin(1e-4)); // no horizontal force
        CHECK(contactForces[1] == Approx(weight).margin(1e-3)); // vertical is weight
        CHECK(contactForces[2] == Approx(0.0).margin(1e-4)); // no horizontal force
        CHECK(contactForces[3] == Approx(0.0).margin(1e-4)); // no torque
        CHECK(contactForces[4] == Approx(0.0).margin(1e-4)); // no torque
        CHECK(contactForces[5] == Approx(0.0).margin(1e-4)); // no torque

        finalHeightTimeStepping =model.getStateVariableValue(state,
            "groundBall/groundBall_coord_2/value");
    }

    // Direct collocation.
    // -------------------
    // This is a simulation (initial value problem), not a trajectory
    // optimization.
    SimTK::Real finalHeightDircol;
    {
        MocoStudy moco;
        MocoProblem& mp = moco.updProblem();
        mp.setModelCopy(model);
        mp.setTimeBounds(0, finalTime);
        mp.setStateInfo("/groundBall/groundBall_coord_0/value", {-1, 1}, 0);
        mp.setStateInfo("/groundBall/groundBall_coord_1/value", {-1, 1}, 0);
        mp.setStateInfo("/groundBall/groundBall_coord_2/value", {-1, 1}, 0.5);
        mp.setStateInfo("/groundBall/groundBall_coord_0/speed", {-10, 10}, 0);
        mp.setStateInfo("/groundBall/groundBall_coord_1/speed", {-10, 10}, 0);
        mp.setStateInfo("/groundBall/groundBall_coord_2/speed", {-10, 10}, 0);

        auto& ms = moco.initCasADiSolver();
        ms.set_num_mesh_points(50);
        ms.set_verbosity(2);
        ms.set_optim_solver("ipopt");

        MocoSolution solution = moco.solve();

        auto statesTraj = solution.exportToStatesTrajectory(mp);
        const auto& finalState = statesTraj.back();
       model.realizeVelocity(finalState);

       auto& contactBallHalfSpace =
           model.getComponent<SmoothSphereHalfSpaceForce>(
               "contactBallHalfSpace");

        Array<double> contactForces =
        contactBallHalfSpace.getRecordValues(finalState);
        CHECK(contactForces[0] == Approx(0.0).margin(1e-4)); // no horizontal force
        CHECK(contactForces[1] == Approx(weight).margin(1e-3)); // vertical is weight
        CHECK(contactForces[2] == Approx(0.0).margin(1e-4)); // no horizontal force
        CHECK(contactForces[3] == Approx(0.0).margin(1e-4)); // no torque
        CHECK(contactForces[4] == Approx(0.0).margin(1e-4)); // no torque
        CHECK(contactForces[5] == Approx(0.0).margin(1e-4)); // no torque

        finalHeightDircol = model.getStateVariableValue(finalState,
            "groundBall/groundBall_coord_2/value");
    }

    CHECK(finalHeightTimeStepping == Approx(finalHeightDircol).margin(1e-5));

    return finalHeightTimeStepping;

}

// TODO this test is not quite right now.
// Test the friction component of the contact force by ensuring that the ball
// travels the expected horizontal distance if it starts in the ground.
// To make the friction force roughly constant, we want the equilibrium height
// of the mass (from testSmoothSphereHalfSpaceForce_NormalForce()).
void testSmoothSphereHalfSpaceForce_FrictionForce(
    const SimTK::Real& equilibriumHeight) {

    Model model(createBallHalfSpaceModel());

    const SimTK::Real y0 = equilibriumHeight;
    const SimTK::Real finalTime = 0.5;
    const SimTK::Real vx0 = 2.5;

    const SimTK::Real g = -model.getGravity()[1];

    // Expected final x position.
    // --------------------------
    // m * vxdot = F = - mu * m * g
    // vx(t) = -mu * g * t + vx0
    // x(t) = -0.5 * mu * g * t^2 + vx0 * t
    // The time at which the point reaches rest:
    // t_rest = vx0 / (mu * g)
    // Final position: x(t_rest)
    const double& mu = FRICTION_COEFFICIENT;
    const SimTK::Real restTime = vx0 / (mu * g);
    CHECK(restTime < finalTime);
    const SimTK::Real expectedFinalX =
            -0.5 * mu * g * pow(restTime, 2) + vx0 * restTime;

    // Time stepping.
    // --------------
    {
        SimTK::State state = model.initSystem();
        model.setStateVariableValue(state,
            "groundBall/groundBall_coord_2/value", y0);
        model.setStateVariableValue(state,
            "groundBall/groundBall_coord_1/speed", vx0);
        Manager manager(model, state);
        state = manager.integrate(finalTime);

        visualize(model, manager.getStateStorage());

        const SimTK::Real finalTX = model.getStateVariableValue(state,
            "groundBall/groundBall_coord_1/value");

        CHECK(finalTX == Approx(expectedFinalX).margin(0.005));

        // The system should be at rest.
        OpenSim_CHECK_MATRIX_ABSTOL(state.getU(),
                SimTK::Vector(state.getNU(), 0.0), 1e-3);

    }

    // Direct collocation.
    // -------------------
    // This is a simulation (initial value problem), not a trajectory
    // optimization.
    {
        MocoStudy moco;
        MocoProblem& mp = moco.updProblem();
        mp.setModelCopy(model);
        mp.setTimeBounds(0, finalTime);
        mp.setStateInfo("/groundBall/groundBall_coord_0/value", {-1, 1}, 0);
        mp.setStateInfo("/groundBall/groundBall_coord_1/value", {-1, 1}, 0);
        mp.setStateInfo("/groundBall/groundBall_coord_2/value", {-1, 1}, y0);
        mp.setStateInfo("/groundBall/groundBall_coord_0/speed", {-10, 10}, 0);
        mp.setStateInfo("/groundBall/groundBall_coord_1/speed", {-10, 10},vx0);
        mp.setStateInfo("/groundBall/groundBall_coord_2/speed", {-10, 10}, 0);

        auto& ms = moco.initCasADiSolver();
        ms.set_num_mesh_points(50);
        ms.set_verbosity(2);
        ms.set_optim_solver("ipopt");

        MocoSolution solution = moco.solve();
        //solution.write("testContact_testFrictionForce_solution.sto");
        //moco.visualize(solution);

        auto statesTraj = solution.exportToStatesTrajectory(mp);
        const auto& finalState = statesTraj.back();
        const SimTK::Real finalTX =
            model.getStateVariableValue(finalState,
            "groundBall/groundBall_coord_1/value");

        CHECK(finalTX == Approx(expectedFinalX).margin(0.005));

        // The system should be at rest.
        OpenSim_CHECK_MATRIX_ABSTOL(finalState.getU(),
                SimTK::Vector(finalState.getNU(), 0.0), 1e-3);
    }
}

AckermannVanDenBogert2010Force* createAVDB() {
    auto* contact = new AckermannVanDenBogert2010Force();
    contact->set_stiffness(1e5);
    contact->set_dissipation(1.0);
    contact->set_friction_coefficient(FRICTION_COEFFICIENT);
    return contact;
}

EspositoMiller2018Force* createEspositoMiller() {
    auto* contact = new EspositoMiller2018Force();
    contact->set_stiffness(1e5);
    contact->set_dissipation(1.0);
    contact->set_friction_coefficient(FRICTION_COEFFICIENT);
    return contact;
}

MeyerFregly2016Force* createMeyerFregly() {
    auto* contact = new MeyerFregly2016Force();
    contact->set_stiffness(1e5);
    contact->set_dissipation(1.0);
    // TODO set friction coefficient.
    return contact;
}

TEST_CASE("testStationPlaneContactForce AckermannVanDenBogert2010Force") {
testStationPlaneContactForce(createAVDB);
}

TEST_CASE("testStationPlaneContactForce EspositoMiller2018Force") {
testStationPlaneContactForce(createEspositoMiller);
}

// TODO does not pass:
// TEST_CASE("testStationPlaneContactForce MeyerFregly2016Force") {
//     testStationPlaneContactForce(createMeyerFregly);
// }


TEST_CASE("testSmoothSphereHalfSpaceForce") {
    const SimTK::Real equilibriumHeight =
        testSmoothSphereHalfSpaceForce_NormalForce();
    // TODO does not pass:
    testSmoothSphereHalfSpaceForce_FrictionForce(equilibriumHeight);
}

