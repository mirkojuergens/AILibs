package hasco.variants.forwarddecomposition;

import java.util.function.Predicate;

import hasco.core.RefinementConfiguredSoftwareConfigurationProblem;
import jaicore.planning.graphgenerators.task.tfd.TFDNode;
import jaicore.search.algorithms.standard.bestfirst.nodeevaluation.INodeEvaluator;
import jaicore.search.problemtransformers.GraphSearchProblemInputToGeneralEvaluatedTraversalTreeViaRDFS;

public class HASCOViaFDAndBestFirstWithRQP extends HASCOViaFDAndBestFirst<Double>{

	public HASCOViaFDAndBestFirstWithRQP(RefinementConfiguredSoftwareConfigurationProblem<Double> configurationProblem, int numSamples, int seed, int timeoutForSingleCompletionEvaluationInMS,
			int timeoutForNodeEvaluationInMS) {
		this(configurationProblem, null, numSamples, seed, timeoutForSingleCompletionEvaluationInMS, timeoutForNodeEvaluationInMS, n -> null);
	}

	public HASCOViaFDAndBestFirstWithRQP(RefinementConfiguredSoftwareConfigurationProblem<Double> configurationProblem, Predicate<TFDNode> prioritingPredicate, int numSamples, int seed, int timeoutForSingleCompletionEvaluationInMS,
			int timeoutForNodeEvaluationInMS, INodeEvaluator<TFDNode, Double> preferredNodeEvaluator) {
		super(configurationProblem, new GraphSearchProblemInputToGeneralEvaluatedTraversalTreeViaRDFS<>(preferredNodeEvaluator, prioritingPredicate, seed, numSamples,
				timeoutForSingleCompletionEvaluationInMS, timeoutForNodeEvaluationInMS));
	}	
	
}
