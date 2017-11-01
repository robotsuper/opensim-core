#ifndef TROPTER_DIRECTCOLLOCATION_HPP
#define TROPTER_DIRECTCOLLOCATION_HPP
// ----------------------------------------------------------------------------
// tropter: DirectCollocation.hpp
// ----------------------------------------------------------------------------
// Copyright (c) 2017 tropter authors
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain a
// copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ----------------------------------------------------------------------------

#include "DirectCollocation.h"
#include "transcription/Trapezoidal.h"
#include <tropter/optimization/SNOPTSolver.h>
#include <tropter/optimization/IPOPTSolver.h>

#include <tropter/Exception.hpp>

namespace tropter {

template<typename T>
DirectCollocationSolver<T>::DirectCollocationSolver(
        std::shared_ptr<const OCProblem> ocproblem,
        const std::string& transcrip,
        const std::string& optsolver,
        const unsigned& num_mesh_points)
        : m_ocproblem(ocproblem)
{
    std::locale loc;
    std::string transcrip_lower = transcrip;
    std::transform(transcrip_lower.begin(), transcrip_lower.end(),
            transcrip_lower.begin(), ::tolower);
    if (transcrip_lower == "trapezoidal") {
        m_transcription.reset(new transcription::Trapezoidal<T>(ocproblem,
                                                             num_mesh_points));
    } else {
        TROPTER_THROW("Unrecognized transcription method %s.", transcrip);
    }

    std::string optsolver_lower = optsolver;
    std::transform(optsolver_lower.begin(), optsolver_lower.end(),
            optsolver_lower.begin(), ::tolower);
    if (optsolver_lower == "ipopt") {
        // TODO this may not be good for IPOPTSolver; IPOPTSolver should
        // have a shared_ptr??
        m_optsolver.reset(new IPOPTSolver(*m_transcription.get()));
    } else if (optsolver_lower == "snopt") {
        m_optsolver.reset(new SNOPTSolver(*m_transcription.get()));
    } else {
        TROPTER_THROW("Unrecognized optimization solver %s.", optsolver);
    }
}

template<typename T>
void DirectCollocationSolver<T>::set_verbosity(int verbosity) {
    TROPTER_VALUECHECK(verbosity == 0 || verbosity == 1,
            "verbosity", verbosity, "0 or 1");
    m_optsolver->set_verbosity(verbosity);
    m_verbosity = verbosity;
}

template<typename T>
OptimalControlSolution DirectCollocationSolver<T>::solve() const
{
    Eigen::VectorXd variables;
    return solve_internal(variables);
}

template<typename T>
OptimalControlSolution DirectCollocationSolver<T>::solve(
        const OptimalControlIterate& initial_guess) const {
    if (initial_guess.empty()) return solve();
    Eigen::VectorXd variables =
            m_transcription->construct_iterate(initial_guess, true);
    return solve_internal(variables);
}

template<typename T>
OptimalControlSolution DirectCollocationSolver<T>::solve_internal(
        Eigen::VectorXd& variables) const {
    double obj_value = m_optsolver->optimize(variables);
    OptimalControlIterate traj =
            m_transcription->deconstruct_iterate(variables);
    OptimalControlSolution solution;
    solution.time = traj.time;
    solution.states = traj.states;
    solution.controls = traj.controls;
    solution.objective = obj_value;
    solution.state_names = traj.state_names;
    solution.control_names = traj.control_names;
    return solution;
}

template<typename T>
void DirectCollocationSolver<T>::print_constraint_values(
        const OptimalControlIterate& ocp_vars, std::ostream& stream) const {
    m_transcription->print_constraint_values(ocp_vars, stream);
}

} // namespace tropter

#endif // TROPTER_DIRECTCOLLOCATION_HPP