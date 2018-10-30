package hasco.knowledgebase.rqp;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.math3.geometry.euclidean.oned.Interval;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import hasco.core.Util;
import hasco.knowledgebase.PerformanceKnowledgeBase;
import hasco.model.CategoricalParameterDomain;
import hasco.model.Component;
import hasco.model.ComponentInstance;
import hasco.model.NumericParameterDomain;
import hasco.model.Parameter;
import jaicore.basic.sets.SetUtil;
import jaicore.logic.fol.structure.Monom;
import jaicore.ml.intervaltree.RangeQueryPredictor;
import jaicore.planning.graphgenerators.IPlanningGraphGeneratorDeriver;
import jaicore.search.algorithms.standard.bestfirst.nodeevaluation.INodeEvaluator;
import jaicore.search.model.travesaltree.Node;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;

/**
 * A node evaluator based on range queries:
 * 
 * @author elppa
 *
 * @param <T>
 * @param <A>
 */
public class RangeQueryBasedNodeEvaluator<T, A> implements INodeEvaluator<T, Double> {

	private static final Logger logger = LoggerFactory.getLogger(RangeQueryBasedNodeEvaluator.class);

	// needed for resolving of the pipeline
	PerformanceKnowledgeBase performanceKnowledgeBase;

	// needed for resolving of the pipeline
	Monom initState;

	// needed for resolving of the pipeline
	Collection<Component> components;

	// needed for resolving of the pipeline
	IPlanningGraphGeneratorDeriver<?, ?, ?, ?, T, A> planningGraphDeriver;

	public RangeQueryBasedNodeEvaluator(PerformanceKnowledgeBase performances) {
		this.performanceKnowledgeBase = performances;
	}

	@Override
	public Double f(Node<T, ?> node) throws Exception {

		// Getting the pipeline node
		ComponentInstance instance = Util.getComponentInstanceForNode(planningGraphDeriver, components, initState, node,
				"RQP", true);

		// Getting the RQP for this pipeline
		RangeQueryPredictor rqp = performanceKnowledgeBase.getTrainedRQPForComposition("test", instance);
		logger.debug("RangeQuery requested for {}", instance);

		// resolving of the range query
		List<Component> allComponents = Util.getComponentsOfComposition(instance);

		Instance rangeQuery = deriveRangeQueryFromSuccessors(allComponents, instance);

		// Query the rqp
		Interval range = rqp.predictInterval(rangeQuery);

		// return solution
		return range.getSup();
	}

	/**
	 * Derives a range query based on the successors of a node.
	 * 
	 * @return
	 */
	final Instance deriveRangeQueryFromSuccessors(List<Component> allComponents, ComponentInstance solution) {
		// sort the list
		Collections.sort(allComponents, (c1, c2) -> c1.getName().compareTo(c2.getName()));
		// calculate the size of the range query (each parameter gets 2 values)
		List<Parameter> allParameters = performanceKnowledgeBase.deriveAllParameters(allComponents);
		Instance instance = new DenseInstance(performanceKnowledgeBase.deriveNumAttributesForRangeQuery(allParameters));
		for (Component component : allComponents) {
			// sort the parameters
			List<Parameter> params = component.getParameters().getTotalOrder();
			for (Parameter p : params) {
				int startIndex = performanceKnowledgeBase.getStartIndexForParameterInRangeQuery(allParameters, p);
				String parameterValue = solution.getParameterValue(p);
				if (p.isCategorical()) {
					deriveCategoricalRange(instance, p, startIndex, parameterValue);
				} else if (p.isNumeric()) {
					deriveNumericRange(instance, startIndex, parameterValue);
				}
			}
		}
		return instance;
	}

	private void deriveNumericRange(Instance instance, int startIndex, String parameterValue) {
		Interval parameterInterval = SetUtil.unserializeInterval(parameterValue);
		instance.setValue(startIndex, parameterInterval.getInf());
		instance.setValue(++startIndex, parameterInterval.getSup());
	}

	private void deriveCategoricalRange(Instance instance, Parameter p, int startIndex, String parameterValue) {
		// manual one-hot-encoding
		int tempIndex = startIndex;
		for (String categoricalValue : ((CategoricalParameterDomain) p.getDefaultDomain()).getValues()) {
			if (categoricalValue.equals(parameterValue)) {
				instance.setValue(tempIndex, 1.0);
			} else {
				instance.setValue(tempIndex, 0.0);
			}
			tempIndex += 1;
		}
	}

	public void setPerformanceKnowledgeBase(PerformanceKnowledgeBase performanceKnowledgeBase) {
		this.performanceKnowledgeBase = performanceKnowledgeBase;
	}

	public void setInitState(Monom initState) {
		this.initState = initState;
	}

	public void setComponents(Collection<Component> components) {
		this.components = components;
	}

	public void setPlanningGraphDeriver(IPlanningGraphGeneratorDeriver<?, ?, ?, ?, T, A> planningGraphDeriver) {
		this.planningGraphDeriver = planningGraphDeriver;
	}

}
