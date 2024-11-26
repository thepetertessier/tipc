#include "Optimizer.h"

#include "llvm/Passes/PassBuilder.h"

#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

// P5 passes
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"

// For logging
#include "loguru.hpp"

namespace { // Anonymous namespace for local function
	    
bool contains(Optimization o, llvm::cl::list<Optimization> &l) {
  for (unsigned i = 0; i != l.size(); ++i) {
    if (o == l[i]) return true;
  }
  return false;
}

}

void Optimizer::optimize(llvm::Module *theModule, 
		llvm::cl::list<Optimization> &enabledOpts) {
  LOG_S(1) << "Optimizing program " << theModule->getName().str();

  // New pass builder
  llvm::PassBuilder passBuilder;

  // Setting-up Analysis Managers for different granularities of optimizations
  llvm::FunctionAnalysisManager functionAnalysisManager;
  llvm::ModuleAnalysisManager moduleAnalysisManager;
  llvm::LoopAnalysisManager loopAnalysisManager;
  llvm::CGSCCAnalysisManager cgsccAnalysisManager;

  // Registering the analysis managers with the pass builder
  passBuilder.registerModuleAnalyses(moduleAnalysisManager);
  passBuilder.registerCGSCCAnalyses(cgsccAnalysisManager);
  passBuilder.registerFunctionAnalyses(functionAnalysisManager);
  passBuilder.registerLoopAnalyses(loopAnalysisManager);

  // Cross Register Proxies so that analysis results can be reused as needed.
  passBuilder.crossRegisterProxies(loopAnalysisManager, functionAnalysisManager,
                                   cgsccAnalysisManager, moduleAnalysisManager);

  // Initiating Function and Module level PassManagers
  llvm::ModulePassManager modulePassManager;
  llvm::FunctionPassManager functionPassManager;
  llvm::LoopPassManager loopPassManagerWithMSSA;
  llvm::LoopPassManager loopPassManager;

  // Adding passes to the pipeline

  // Constructs SSA and is a pre-requisite for many other passes
  functionPassManager.addPass(llvm::PromotePass());

  // Instruction combine pass scans for a variety of patterns and replaces bitcodes matched with improvements.
  functionPassManager.addPass(llvm::InstCombinePass());

  // Reassociate expressions.
  functionPassManager.addPass(llvm::ReassociatePass());

  // Eliminate Common SubExpressions using the Global Value Numbering (GVN) algorithm.
  functionPassManager.addPass(llvm::GVNPass());

  // Simplify the control flow graph (deleting unreachable blocks, etc).
  functionPassManager.addPass(llvm::SimplifyCFGPass());

  if (contains(licm, enabledOpts)) {
    // Add loop invariant code motion 
    loopPassManagerWithMSSA.addPass(llvm::LICMPass(llvm::LICMOptions())); 
  }

  if (contains(del, enabledOpts)) {
    // Add loop deletion pass
    loopPassManager.addPass(llvm::LoopDeletionPass()); 
  }   

  // Add loop pass managers with and w/out MemorySSA
  functionPassManager.addPass(
      createFunctionToLoopPassAdaptor(std::move(loopPassManagerWithMSSA),true));

  functionPassManager.addPass(
      createFunctionToLoopPassAdaptor(std::move(loopPassManager)));

  // Passing the function pass manager to the modulePassManager using a function
  // adaptor, then passing theModule to the ModulePassManager along with
  // ModuleAnalysisManager.
  modulePassManager.addPass(
      createModuleToFunctionPassAdaptor(std::move(functionPassManager), true));


  if (contains(fin, enabledOpts)) {
    modulePassManager.addPass(llvm::ModuleInlinerPass());
  }

  modulePassManager.run(*theModule, moduleAnalysisManager);
}
