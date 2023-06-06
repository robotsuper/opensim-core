/* -------------------------------------------------------------------------- *
 * OpenSim Moco: sandboxSandbox.cpp                                           *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2019 Stanford University and the Authors                     *
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

// This file provides a way to easily prototype or test temporary snippets of
// code during development.

#include <OpenSim/Moco/osimMoco.h>
#include <iostream>
#include <string>

using namespace OpenSim;

int main() {
    std::string baselineSimulationName = "SIM_baseline_20230413T135455";
    std::string baselineSolutionFile = baselineSimulationName + "_solution.sto";
    std::string baselineModelFile = baselineSimulationName + "_model.osim";
    std::string baselineOutputsFile = baselineSimulationName + "_outputs.sto";
    std::string baselineGRFFile = baselineSimulationName + "_GRF.sto";
    MocoTrajectory baselineTrajectory(baselineSolutionFile);
    TimeSeriesTable baselineStates = baselineTrajectory.exportToStatesTable();
    TimeSeriesTable baselineControls = baselineTrajectory.exportToControlsTable();
    double initialTime = baselineStates.getIndependentColumn()[0];
    double finalTime = baselineStates.getIndependentColumn()[baselineStates.getNumRows()-1];

    Model model(baselineModelFile);

    MocoStudy study;
    MocoProblem& problem = study.updProblem();
    double finalTimeUpperBound = finalTime + 0.03;
    double finalTimeLowerBound = finalTime - 0.03;
    double initialTimeUpperBound = initialTime + 0.03;
    double initialTimeLowerBound = initialTime - 0.03;
    problem.setTimeBounds(MocoInitialBounds(initialTimeLowerBound,initialTimeUpperBound), MocoFinalBounds(finalTimeLowerBound,finalTimeUpperBound));
    problem.setModel(std::make_unique<Model>(model));

    auto* stateTrackingGoal = problem.addGoal<MocoStateTrackingGoal>();
    stateTrackingGoal->setName("baseline_state_tracking");
    stateTrackingGoal->setAllowUnusedReferences(false);
    stateTrackingGoal->setScaleWeightsWithRange(false);
    stateTrackingGoal->setReference(TableProcessor(baselineStates));

    auto* controlTrackingGoal = problem.addGoal<MocoControlTrackingGoal>();
    controlTrackingGoal->setName("baseline_control_tracking");
    controlTrackingGoal->setAllowUnusedReferences(false);
    controlTrackingGoal->setReference(TableProcessor(baselineControls));

    double percentChangeEccentricWork = -0.025;
    double percentChangeEccentricWorkBound = 0.005;
    double percentChangeEccentricWorkLowerBound = percentChangeEccentricWork - percentChangeEccentricWorkBound;
    double percentChangeEccentricWorkUpperBound = percentChangeEccentricWork + percentChangeEccentricWorkBound;

    auto* controlGoal = problem.addGoal<MocoControlGoal>();
    controlGoal->setName("control_effort");
//    controlGoal->setMode("endpoint_constraint");

    std::string thisHamstring = "bflh_r";
//    auto* thisJHSI = problem.addGoal<MocoOutputExtremumGoal>();
//    thisJHSI->setName(thisHamstring + "_eccentric_work");
//    thisJHSI->setExponent(1);
//    thisJHSI->setDivideByDisplacement(false);
//    thisJHSI->setExtremumType("minimum");
//    thisJHSI->setDivideByMass(false);
//    thisJHSI->setSmoothingFactor(0.2);
//    thisJHSI->setOutputPath({"/forceset/" + thisHamstring + "|muscle_power"});
//    thisJHSI->setEnabled(true);
//    thisJHSI->setMode("endpoint_constraint");
//    double baselineHamstringEccentricWork = -93.1435;
//    double thisEccentricWorkLowerBound = (1-percentChangeEccentricWorkLowerBound) * baselineHamstringEccentricWork;
//    double thisEccentricWorkUpperBound = (1-percentChangeEccentricWorkUpperBound) * baselineHamstringEccentricWork;
//    std::vector<MocoBounds> thisConstraintBounds;
//    thisConstraintBounds.emplace_back(thisEccentricWorkLowerBound,thisEccentricWorkUpperBound);
//    thisJHSI->setEndpointConstraintBounds(thisConstraintBounds);

    MocoCasADiSolver& solver = study.initCasADiSolver();
    solver.set_num_mesh_intervals(5);
    solver.set_parallel(1);

    study.solve();

}
