package hasco.variants.forwarddecomposition;
import java.util.function.Predicate;

import hasco.core.DefaultHASCOPlanningGraphGeneratorDeriver;
import hasco.core.HASCO;
import hasco.core.HASCOFactory;
import hasco.knowledgebase.PerformanceKnowledgeBase;
import hasco.knowledgebase.PerformanceSampleListener;
import hasco.knowledgebase.rqp.RangeQueryBasedNodeEvaluator;
import jaicore.planning.algorithms.forwarddecomposition.ForwardDecompositionReducer;
import jaicore.planning.graphgenerators.task.tfd.TFDNode;
import jaicore.search.algorithms.standard.bestfirst.BestFirstFactory;
import jaicore.search.model.probleminputs.GeneralEvaluatedTraversalTree;
import jaicore.search.problemtransformers.GraphSearchProblemInputToGeneralEvaluatedTraversalTreeViaRDFS;

public class HASCOViaFDAndBestFirstWithRQPFactory extends HASCOFactory<GeneralEvaluatedTraversalTree<TFDNode, String, Double>, TFDNode, String, Double>{

	private PerformanceKnowledgeBase performanceKnowledgeBase;
	
	private RangeQueryBasedNodeEvaluator<TFDNode, String> rangeQueryBasedNodeEvaluator = new RangeQueryBasedNodeEvaluator<>(performanceKnowledgeBase);
	
	private Predicate<TFDNode> priorizingPredicate;
	
	public HASCOViaFDAndBestFirstWithRQPFactory() {
		super();
		setSearchFactory(new BestFirstFactory<>());
		setSearchProblemTransformer(new GraphSearchProblemInputToGeneralEvaluatedTraversalTreeViaRDFS<>(rangeQueryBasedNodeEvaluator, priorizingPredicate, 1, 3, -1, -1));
		setPlanningGraphGeneratorDeriver(new DefaultHASCOPlanningGraphGeneratorDeriver<>(new ForwardDecompositionReducer<>()));
	}

	@Override
	public HASCO<GeneralEvaluatedTraversalTree<TFDNode, String, Double>, TFDNode, String, Double> getAlgorithm() {
		 HASCO<GeneralEvaluatedTraversalTree<TFDNode, String, Double>, TFDNode, String, Double> hasco =  super.getAlgorithm();
		 hasco.registerListener(new PerformanceSampleListener(performanceKnowledgeBase, "test"));
		 rangeQueryBasedNodeEvaluator.setComponents(problem.getComponents());
		 rangeQueryBasedNodeEvaluator.setPlanningGraphDeriver(planningGraphGeneratorDeriver);
		 rangeQueryBasedNodeEvaluator.setInitState(hasco.getPlanningProblem().getCorePlanningProblem().getInit());
		 return hasco;
	}
	
	
	
	public Predicate<TFDNode> getPriorizingPredicate() {
		return priorizingPredicate;
	}

	public void setPriorizingPredicate(Predicate<TFDNode> priorizingPredicate) {
		this.priorizingPredicate = priorizingPredicate;
	}
}
