package hasco.variants.forwarddecomposition;

import hasco.core.RefinementConfiguredSoftwareConfigurationProblem;
import hasco.knowledgebase.rqp.RangeQueryBasedNodeEvaluator;
import jaicore.basic.algorithm.AlgorithmInitializedEvent;
import jaicore.basic.algorithm.AlgorithmProblemTransformer;
import jaicore.planning.graphgenerators.task.tfd.TFDNode;
import jaicore.search.model.probleminputs.GeneralEvaluatedTraversalTree;
import jaicore.search.model.probleminputs.GraphSearchProblemInput;

public class HASCOViaFDAndBestFirstWithRQP extends HASCOViaFDAndBestFirst<Double> {

	protected RangeQueryBasedNodeEvaluator<TFDNode, ?> nodeEvaluator;

	public HASCOViaFDAndBestFirstWithRQP(RefinementConfiguredSoftwareConfigurationProblem<Double> configurationProblem,
			AlgorithmProblemTransformer<GraphSearchProblemInput<TFDNode, String, Double>, GeneralEvaluatedTraversalTree<TFDNode, String, Double>> searchProblemTransformer,
			RangeQueryBasedNodeEvaluator<TFDNode, ?> nodeEvaluator) {
		super(configurationProblem, searchProblemTransformer);
		this.nodeEvaluator = nodeEvaluator;
	}

	@Override
	public AlgorithmInitializedEvent init() {
		AlgorithmInitializedEvent aiEvent = super.init();
		nodeEvaluator.setInitState(this.getPlanningProblem().getCorePlanningProblem().getInit());
		return aiEvent;
	}

	
}
