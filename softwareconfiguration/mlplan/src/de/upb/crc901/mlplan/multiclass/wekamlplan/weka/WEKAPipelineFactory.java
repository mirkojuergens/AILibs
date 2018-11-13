package de.upb.crc901.mlplan.multiclass.wekamlplan.weka;

import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;

import de.upb.crc901.mlplan.multiclass.wekamlplan.ClassifierFactory;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import hasco.model.ComponentInstance;
import hasco.serialization.CompositionSerializer;
import jaicore.basic.ListHelper;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;

public class WEKAPipelineFactory implements ClassifierFactory {

	@Override
	public MLPipeline getComponentInstantiation(final ComponentInstance groundComponent) throws Exception {

		ComponentInstance preprocessorCI = null;
		String ppName = "";
		ComponentInstance classifierCI = null;
		System.out.println(new CompositionSerializer().serializeComponentInstance(groundComponent));

		switch (groundComponent.getComponent().getName()) {
		case "pipeline": {
			/* Retrieve component instances of pipeline */
			preprocessorCI = groundComponent.getSatisfactionOfRequiredInterfaces().get("preprocessor");
			ppName = preprocessorCI.getComponent().getName();

			classifierCI = groundComponent.getSatisfactionOfRequiredInterfaces().get("classifier");
			break;
		}
		default: {
			classifierCI = groundComponent;
			break;
		}
		}

		ASEvaluation eval = null;
		ASSearch search = null;
		if (ppName.startsWith("weka")) {
			ComponentInstance evaluatorCI = preprocessorCI.getSatisfactionOfRequiredInterfaces().get("eval");
			ComponentInstance searcherCI = preprocessorCI.getSatisfactionOfRequiredInterfaces().get("search");

			eval = ASEvaluation.forName(evaluatorCI.getComponent().getName(), this.getParameterList(evaluatorCI).toArray(new String[] {}));
			search = ASSearch.forName(searcherCI.getComponent().getName(), this.getParameterList(searcherCI).toArray(new String[] {}));
		}

		classifierCI.getParameterValues();
		List<String> parameters = this.getParameterList(classifierCI);
		Classifier c = AbstractClassifier.forName(classifierCI.getComponent().getName(), parameters.toArray(new String[] {}));
		// System.out.println(((search != null) ? search.getClass().getName() : "") + "
		// "
		// + ((eval != null) ? eval.getClass().getName() : "") + " " +
		// c.getClass().getName());
		return new MLPipeline(search, eval, c);
	}

	private List<String> getParameterList(final ComponentInstance ci) {
		List<String> parameters = new LinkedList<>();

		for (Entry<String, String> parameterValues : ci.getParameterValues().entrySet()) {
			if (parameterValues.getKey().toLowerCase().endsWith("activator") || parameterValues.getValue().equals("REMOVED")) {
				continue;
			}

			if (!parameterValues.getValue().equals("false")) {
				parameters.add("-" + parameterValues.getKey());
			}
			if (parameterValues.getValue() != null && !parameterValues.getValue().equals("") && !parameterValues.getValue().equals("true") && !parameterValues.getValue().equals("false")) {
				parameters.add(parameterValues.getValue());
			}
		}

		for (String paramName : ci.getSatisfactionOfRequiredInterfaces().keySet()) {
			List<String> subParams = this.getParameterList(ci.getSatisfactionOfRequiredInterfaces().get(paramName));
			String paramValue = ci.getSatisfactionOfRequiredInterfaces().get(paramName).getComponent().getName() + " " + ListHelper.implode(subParams, " ");
			parameters.add("-" + paramName);
			parameters.add(paramValue);
		}
		return parameters;
	}

}
