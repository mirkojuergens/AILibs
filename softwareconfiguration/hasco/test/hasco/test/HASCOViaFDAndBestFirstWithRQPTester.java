package hasco.test;

import org.junit.AfterClass;

import hasco.core.HASCOFactory;
import hasco.variants.forwarddecomposition.HASCOViaFDAndBestFirstWithRandomCompletionsFactory;
import jaicore.planning.graphgenerators.task.tfd.TFDNode;
import jaicore.search.model.probleminputs.GeneralEvaluatedTraversalTree;

public class HASCOViaFDAndBestFirstWithRQPTester extends HASCOTester<GeneralEvaluatedTraversalTree<TFDNode, String, Double>, TFDNode, String> {

	@Override
	public HASCOFactory<GeneralEvaluatedTraversalTree<TFDNode, String, Double>, TFDNode, String, Double> getFactory() {
		HASCOViaFDAndBestFirstWithRandomCompletionsFactory factory = new HASCOViaFDAndBestFirstWithRandomCompletionsFactory();
		factory.setVisualizationEnabled(true);
		return factory;
	}
}
