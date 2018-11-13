package hasco.knowledgebase;

import java.io.IOException;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.stream.Collectors;

import org.apache.commons.lang3.tuple.Pair;
<<<<<<< HEAD
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
=======
import org.jfree.util.Log;
>>>>>>> jonas/dev

import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.module.SimpleModule;

import hasco.core.Util;
import hasco.model.CategoricalParameterDomain;
import hasco.model.Component;
import hasco.model.ComponentInstance;
import hasco.model.Dependency;
import hasco.model.NumericParameterDomain;
import hasco.model.Parameter;
import hasco.model.ParameterDomain;
import hasco.serialization.ParameterDeserializer;
import hasco.serialization.ParameterDomainDeserializer;
import jaicore.basic.SQLAdapter;
import jaicore.basic.sets.PartialOrderedSet;
import jaicore.ml.core.FeatureSpace;
import jaicore.ml.intervaltree.ExtendedRandomForest;
import jaicore.ml.intervaltree.RangeQueryPredictor;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.ProtectedProperties;

/**
 * Knowledge base that manages observed performance behavior
 * 
 * @author jmhansel
 *
 */

public class PerformanceKnowledgeBase {

	private static final Logger log = LoggerFactory.getLogger(PerformanceKnowledgeBase.class);

	private SQLAdapter sqlAdapter;
	private Map<String, HashMap<ComponentInstance, Double>> performanceSamples;
	/** This is map contains a String */
	private Map<String, HashMap<String, List<Pair<ParameterConfiguration, Double>>>> performanceSamplesByIdentifier;
	private Map<String, HashMap<String, Instances>> performanceInstancesByIdentifier;
	private Map<String, HashMap<String, Instances>> performanceInstancesIndividualComponents;

<<<<<<< HEAD
=======
	/**
	 * Inner helper class for managing parameter configurations easily.
	 * 
	 * @author jmhansel
	 *
	 */
	private class ParameterConfiguration {
		private final List<Pair<Parameter, String>> values;

		public ParameterConfiguration(ComponentInstance composition) {
			ArrayList<Pair<Parameter, String>> temp = new ArrayList<Pair<Parameter, String>>();
			List<ComponentInstance> componentInstances = Util.getComponentInstancesOfComposition(composition);
			for (ComponentInstance compInst : componentInstances) {
				List<Parameter> parameters = compInst.getComponent().getParameters().getTotalOrder();
				for (Parameter parameter : parameters) {
					// TODO check if this is feasible
					String value;
					if (compInst.getParametersThatHaveBeenSetExplicitly().contains(parameter)) {
						value = compInst.getParameterValues().get(parameter.getName());
					} else {
						value = parameter.getDefaultValue().toString();
					}
					temp.add(Pair.of(parameter, value));
				}
			}
			// Make the list immutable to avoid problems with hashCode
			values = Collections.unmodifiableList(temp);
		}

		@Override
		public int hashCode() {
			return values.hashCode();
		}

		public List<Pair<Parameter, String>> getValues() {
			return this.values;
		}
	}

>>>>>>> jonas/dev
	public PerformanceKnowledgeBase(final SQLAdapter sqlAdapter) {
		this();
		this.sqlAdapter = sqlAdapter;
<<<<<<< HEAD
=======
		this.performanceInstancesByIdentifier = new HashMap<String, HashMap<String, Instances>>();
		this.performanceInstancesIndividualComponents = new HashMap<String, HashMap<String, Instances>>();
>>>>>>> jonas/dev
	}

	public PerformanceKnowledgeBase() {
		super();
<<<<<<< HEAD
		this.performanceInstancesByIdentifier = new HashMap<>();
		this.performanceInstancesIndividualComponents = new HashMap<>();
=======
		this.performanceInstancesByIdentifier = new HashMap<String, HashMap<String, Instances>>();
		this.performanceInstancesIndividualComponents = new HashMap<String, HashMap<String, Instances>>();
>>>>>>> jonas/dev
	}

	/**
	 * Adds a performance sample to the knowledge base, if it does not already
	 * exist.
	 * 
	 * @param benchmarkName
	 *            the dataset +
	 * @param componentInstance
	 *            the hasco instance
	 * @param score
	 *            performance score
	 * @param addToDB
	 */
	public void addPerformanceSample(String benchmarkName, ComponentInstance componentInstance, double score,
			boolean addToDB) {
		String identifier = Util.getComponentNamesOfComposition(componentInstance);

		if (performanceInstancesByIdentifier.get(benchmarkName) == null) {
<<<<<<< HEAD
			log.debug("First performance score for this component instance -> Creating new HashMap");
			performanceInstancesByIdentifier.put(benchmarkName, new HashMap<>());
			performanceInstancesIndividualComponents.put(benchmarkName, new HashMap<>());
		}

		if (!performanceInstancesByIdentifier.get(benchmarkName).containsKey(identifier)) {
			createNewInstancesForBenchmark(benchmarkName, componentInstance, identifier);
=======
			HashMap<String, Instances> newMap = new HashMap<String, Instances>();
			HashMap<String, Instances> newMap2 = new HashMap<String, Instances>();
			performanceInstancesByIdentifier.put(benchmarkName, newMap);
			performanceInstancesIndividualComponents.put(benchmarkName, newMap2);

		}

		if (!performanceInstancesByIdentifier.get(benchmarkName).containsKey(identifier)) {
			// Create Instances pipeline for this pipeline type
			Instances instances = null;
			// Add parameter domains as attributes
			List<ComponentInstance> componentInstances = Util.getComponentInstancesOfComposition(componentInstance);
			ArrayList<Attribute> allAttributes = new ArrayList<Attribute>();
			for (ComponentInstance ci : componentInstances) {
				List<Parameter> parameters = ci.getComponent().getParameters().getTotalOrder();
				ArrayList<Attribute> attributes = new ArrayList<Attribute>(parameters.size());
				for (Parameter parameter : parameters) {
					ParameterDomain domain = parameter.getDefaultDomain();
					Attribute attr = null;
					if (domain instanceof CategoricalParameterDomain) {
						CategoricalParameterDomain catDomain = (CategoricalParameterDomain) domain;
						// TODO further namespacing of attributes!!!
						attr = new Attribute(ci.getComponent().getName() + "::" + parameter.getName(),
								Arrays.asList(catDomain.getValues()));
					} else if (domain instanceof NumericParameterDomain) {
						NumericParameterDomain numDomain = (NumericParameterDomain) domain;
						// TODO is there a better way to set the range of this attribute?
						String range = "[" + numDomain.getMin() + "," + numDomain.getMax() + "]";
						Properties prop = new Properties();
						prop.setProperty("range", range);
						ProtectedProperties metaInfo = new ProtectedProperties(prop);
						attr = new Attribute(ci.getComponent().getName() + "::" + parameter.getName(), metaInfo);
					}

					attributes.add(attr);
				}
				allAttributes.addAll(attributes);
			}
			// Add performance score as class attribute TODO make score numeric?
			Attribute scoreAttr = new Attribute("performance_score");
			allAttributes.add(scoreAttr);
			instances = new Instances("performance_samples", allAttributes, 16);
			instances.setClass(scoreAttr);
			performanceInstancesByIdentifier.get(benchmarkName).put(identifier, instances);
>>>>>>> jonas/dev
		}
		// TODO Test this
		List<ComponentInstance> componentInstances = Util.getComponentInstancesOfComposition(componentInstance);
		for (ComponentInstance ci : componentInstances) {
			if (!performanceInstancesIndividualComponents.get(benchmarkName).containsKey(ci.getComponent().getName())) {
				// Create Instances pipeline for this pipeline type
				Instances instances = null;
				// Add parameter domains as attributes
				List<Parameter> parameters = ci.getComponent().getParameters().getTotalOrder();
				ArrayList<Attribute> attributes = new ArrayList<Attribute>(parameters.size());
				for (Parameter parameter : parameters) {
					ParameterDomain domain = parameter.getDefaultDomain();
					Attribute attr = null;
					if (domain instanceof CategoricalParameterDomain) {
						CategoricalParameterDomain catDomain = (CategoricalParameterDomain) domain;
						// TODO further namespacing of attributes!!!
						attr = new Attribute(parameter.getName(), Arrays.asList(catDomain.getValues()));
					} else if (domain instanceof NumericParameterDomain) {
						NumericParameterDomain numDomain = (NumericParameterDomain) domain;
						// TODO is there a better way to set the range of this attribute?
						String range = "[" + numDomain.getMin() + "," + numDomain.getMax() + "]";
						Properties prop = new Properties();
						prop.setProperty("range", range);
						ProtectedProperties metaInfo = new ProtectedProperties(prop);
						attr = new Attribute(parameter.getName(), metaInfo);
					}

					attributes.add(attr);
				}
				Attribute scoreAttr = new Attribute("performance_score");
				attributes.add(scoreAttr);
				instances = new Instances("performance_samples", attributes, 16);
				instances.setClass(scoreAttr);
				performanceInstancesIndividualComponents.get(benchmarkName).put(ci.getComponent().getName(), instances);
			}
		}

		// Add Instance for performance samples to corresponding Instances
		Instances instances = performanceInstancesByIdentifier.get(benchmarkName).get(identifier);
		DenseInstance instance = new DenseInstance(instances.numAttributes());
		ParameterConfiguration config = new ParameterConfiguration(componentInstance);
		List<Pair<Parameter, String>> values = config.getValues();
		for (int i = 0; i < instances.numAttributes() - 1; i++) {
			Attribute attr = instances.attribute(i);
			Parameter param = values.get(i).getLeft();
<<<<<<< HEAD
			if (param.isCategorical()) {
				String value = values.get(i).getRight();
				instance.setValue(attr, value);
			} else if (param.isNumeric()) {
				double finalValue = Double.parseDouble(values.get(i).getRight());
				instance.setValue(attr, finalValue);
=======
			if (values.get(i).getRight() != null) {
				if (param.isCategorical()) {
					String value = values.get(i).getRight();
					boolean attrContainsValue = false;
					Enumeration<Object> possibleValues = attr.enumerateValues();
					while (possibleValues.hasMoreElements() && !attrContainsValue) {
						Object o = possibleValues.nextElement();
						if (o.equals(value))
							attrContainsValue = true;
					}
					if (attrContainsValue)
						instance.setValue(attr, value);
					else
						Log.info("The value you're trying to insert is not in the attributes range!");
				} else if (param.isNumeric()) {
					double finalValue = Double.parseDouble(values.get(i).getRight());
					instance.setValue(attr, finalValue);
				}
>>>>>>> jonas/dev
			}
		}
		Attribute scoreAttr = instances.classAttribute();
		instance.setValue(scoreAttr, score);
		performanceInstancesByIdentifier.get(benchmarkName).get(identifier).add(instance);


		// Add Instance for individual component
		for (ComponentInstance ci : componentInstances) {
			Instances instancesInd = performanceInstancesIndividualComponents.get(benchmarkName)
					.get(ci.getComponent().getName());
			DenseInstance instanceInd = new DenseInstance(instancesInd.numAttributes());
			for (int i = 0; i < instancesInd.numAttributes() - 1; i++) {
				Attribute attr = instancesInd.attribute(i);
				Parameter param = ci.getComponent().getParameterWithName(attr.name());
				String value;
				if (ci.getParametersThatHaveBeenSetExplicitly().contains(param))
					value = ci.getParameterValues().get(param.getName());
				else
					value = param.getDefaultValue().toString();
				if (value != null) {
					if (param.isCategorical()) {
						boolean attrContainsValue = false;
						Enumeration<Object> possibleValues = attr.enumerateValues();
						while (possibleValues.hasMoreElements() && !attrContainsValue) {
							Object o = possibleValues.nextElement();
							if (o.equals(value))
								attrContainsValue = true;
						}
						if (attrContainsValue)
							instanceInd.setValue(attr, value);
					} else if (param.isNumeric()) {
						double finalValue = Double.parseDouble(value);
						instanceInd.setValue(attr, finalValue);
					}
				}
			}
			Attribute scoreAttrInd = instancesInd.classAttribute();
			instanceInd.setValue(scoreAttrInd, score);
			performanceInstancesIndividualComponents.get(benchmarkName).get(ci.getComponent().getName())
					.add(instanceInd);

		}

		if (addToDB)
			this.addPerformanceSampleToDB(benchmarkName, componentInstance, score);
	}

	private void createNewInstancesForBenchmark(String benchmarkName, ComponentInstance componentInstance,
			String identifier) {
		log.debug("Creating new Instances Object");
		// Create Instances pipeline for this pipeline type
		Instances instances = null;
		// Add parameter domains as attributes
		List<ComponentInstance> componentInstances = Util.getComponentInstancesOfComposition(componentInstance);
		ArrayList<Attribute> allAttributes = new ArrayList<>();
		for (ComponentInstance ci : componentInstances) {
			PartialOrderedSet<Parameter> parameters = ci.getComponent().getParameters();
			ArrayList<Attribute> attributes = new ArrayList<>(parameters.size());
			for (Parameter parameter : parameters) {
				ParameterDomain domain = parameter.getDefaultDomain();
				Attribute attr = null;
				if (domain instanceof CategoricalParameterDomain) {
					CategoricalParameterDomain catDomain = (CategoricalParameterDomain) domain;
					attr = new Attribute(getAttributeName(ci, parameter), Arrays.asList(catDomain.getValues()));
				} else if (domain instanceof NumericParameterDomain) {
					NumericParameterDomain numDomain = (NumericParameterDomain) domain;
					// TODO is there a better way to set the range of this attribute?
					String range = "[" + numDomain.getMin() + "," + numDomain.getMax() + "]";
					Properties prop = new Properties();
					prop.setProperty("range", range);
					ProtectedProperties metaInfo = new ProtectedProperties(prop);
					attr = new Attribute(getAttributeName(ci, parameter), metaInfo);
				}
				// System.out.println("Trying to add parameter: " + attr.name() + " for
				// component: "
				// + componentInstance.getComponent().getName());

				attributes.add(attr);
			}
			allAttributes.addAll(attributes);
		}
		// Add performance score as class attribute TODO make score numeric?
		Attribute scoreAttr = new Attribute("performance_score");
		allAttributes.add(scoreAttr);
		instances = new Instances("performance_samples", allAttributes, 16);
		instances.setClass(scoreAttr);
		performanceInstancesByIdentifier.get(benchmarkName).put(identifier, instances);
	}

	private String getAttributeName(ComponentInstance ci, Parameter parameter) {
		// TODO further namespacing of attributes!!!
		return ci.getComponent().getName() + "::" + parameter.getName();
	}

	public Map<String, HashMap<String, Instances>> getPerformanceSamples() {
<<<<<<< HEAD
		// return this.performanceInstancesByIdentifier;
=======
>>>>>>> jonas/dev
		return this.performanceInstancesIndividualComponents;
	}

	public Map<String, HashMap<String, List<Pair<ParameterConfiguration, Double>>>> getPerformanceSamplesByIdentifier() {
		return performanceSamplesByIdentifier;
	}

	public String getStringOfMaps() {
		return performanceSamples.toString();
	}

	public FeatureSpace createFeatureSpaceFromComponentInstance(ComponentInstance compInst) {
		FeatureSpace space = new FeatureSpace();
		List<Parameter> parameters = compInst.getComponent().getParameters().getTotalOrder();
		for (Parameter param : parameters) {
			ParameterDomain domain = param.getDefaultDomain();
		}
		return space;
	}

	public void initializeDBTables() {
		/* initialize tables if not existent */
		try {
			ResultSet rs = sqlAdapter.getResultsOfQuery("SHOW TABLES");
			boolean havePerformanceTable = false;
			while (rs.next()) {
				String tableName = rs.getString(1);
				if (tableName.equals("performance_samples_J48")) {
					havePerformanceTable = true;
				}
			}

			if (!havePerformanceTable) {
				System.out.println("Creating table for performance samples");
				sqlAdapter.update("CREATE TABLE `performance_samples_J48` (\r\n"
						+ " `sample_id` int(10) NOT NULL AUTO_INCREMENT,\r\n"
						+ " `dataset` varchar(200) COLLATE utf8_bin DEFAULT NULL,\r\n"
						+ " `composition` json NOT NULL,\r\n" + " `error_rate` double NOT NULL,\r\n"
						+ " `test_evaluation_technique` varchar(20) ,\r\n" + " `test_split_technique` varchar(20) ,\r\n"
						+ " `val_evaluation_technique` varchar(20) ,\r\n" + " `val_split_technique` varchar(20) ,\r\n"
						+ " `test_seed` int(11) ,\r\n" + " `val_seed` int(11) ,\r\n" + " PRIMARY KEY (`sample_id`)\r\n"
						+ ") ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8 COLLATE=utf8_bin", new ArrayList<>());
			}

		} catch (SQLException e) {
			e.printStackTrace();
		}
	}

	public void addPerformanceSampleToDB(String benchmarkName, ComponentInstance componentInstance, double score) {
		try {
			Map<String, String> map = new HashMap<>();
			map.put("benchmark", benchmarkName);
			ObjectMapper mapper = new ObjectMapper();
			String composition = mapper.writeValueAsString(componentInstance);
			map.put("composition", composition);
			map.put("score", "" + score);
			this.sqlAdapter.insert("performance_samples", map);
		} catch (Throwable e) {
			e.printStackTrace();
		}
	}

	/**
	 * Returns the number of samples for the given benchmark name and pipeline
	 * identifier.
	 * 
	 * @param benchmarkName
	 * @param identifier
	 * @return
	 */
	public int getNumSamples(String benchmarkName, String identifier) {
		if (!this.performanceInstancesByIdentifier.containsKey(benchmarkName))
			return 0;
		if (!this.performanceInstancesByIdentifier.get(benchmarkName).containsKey(identifier))
			return 0;

		return this.performanceInstancesByIdentifier.get(benchmarkName).get(identifier).numInstances();
	}

	/**
	 * Returns the number of significant samples for the given benchmark name and
	 * pipeline identifier. Significant means, that
	 * 
	 * @param benchmarkName
	 * @param identifier
	 * @return
	 */
	public int getNumSignificantSamples(String benchmarkName, String identifier) {
		if (!this.performanceInstancesByIdentifier.containsKey(benchmarkName))
			return 0;
		if (!this.performanceInstancesByIdentifier.get(benchmarkName).containsKey(identifier))
			return 0;
		Instances instances = this.performanceInstancesByIdentifier.get(benchmarkName).get(identifier);
		int numDistinctValues = 1;
		for (int i = 0; i < instances.numInstances(); i++) {
			for (int j = 0; j < i; j++) {
				boolean allValuesDistinct = true;
				for (int k = 0; k < instances.numAttributes(); k++) {
					if (instances.get(i).value(k) == instances.get(j).value(k)) {
						allValuesDistinct = false;
					}
				}
				if (allValuesDistinct)
					numDistinctValues++;
			}
		}
		return numDistinctValues;
	}

	public void loadPerformanceSamplesFromDB() {
		if (sqlAdapter == null) {
			System.out.println("please set an SQL adapter");
			return;
		}
		try {
			ResultSet rs = sqlAdapter
					.getResultsOfQuery("SELECT dataset, composition, error_rate FROM performance_samples_J48");
			ObjectMapper mapper = new ObjectMapper();
			while (rs.next()) {
				String benchmarkName = rs.getString(1);
				String ciString = rs.getString(2);
				if (!benchmarkName.equals("test")) {
					SimpleModule parameterModule = new SimpleModule();
					ParameterDeserializer des = new ParameterDeserializer();
					parameterModule.addDeserializer(Parameter.class, des);

					SimpleModule parameterDomainModule = new SimpleModule();
					ParameterDomainDeserializer parameterDomainDes = new ParameterDomainDeserializer();
					parameterDomainModule.addDeserializer(Dependency.class, parameterDomainDes);

					// mapper.registerModule(parameterModule);
					// mapper.registerModule(parameterDomainModule);

					ComponentInstance composition = mapper.readValue(ciString, ComponentInstance.class);
					double score = rs.getDouble(3);
					this.addPerformanceSample(benchmarkName, composition, score, false);
				}
			}
		} catch (SQLException e) {
			e.printStackTrace();
		} catch (JsonParseException e) {
			e.printStackTrace();
		} catch (JsonMappingException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Checks whether k samples are available, which are
	 * 
	 * @param k
	 * @return
	 */
	public boolean kDistinctAttributeValuesAvailable(String benchmarkName, ComponentInstance composition, int minNum) {
		String identifier = Util.getComponentNamesOfComposition(composition);
		if (!this.performanceInstancesByIdentifier.containsKey(benchmarkName))
			return false;
		if (!this.performanceInstancesByIdentifier.get(benchmarkName).containsKey(identifier))
			return false;
		Instances instances = performanceInstancesByIdentifier.get(benchmarkName).get(identifier);
		if (instances.numInstances() < minNum)
			return false;
		for (int i = 0; i < instances.numAttributes() - 1; i++) {
			// if the attribute is nominal or string but the number of values is smaller
			// than k, skip it
			if (instances.attribute(i).numValues() > 0 && instances.attribute(i).numValues() < minNum) {
				if (instances.numDistinctValues(i) < instances.attribute(i).numValues())
					return false;
			} else if (instances.attribute(i).getUpperNumericBound() <= instances.attribute(i).getLowerNumericBound()) {
				continue;
			} else if (instances.numDistinctValues(i) < minNum) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Checks whether at least k sample are available, that are pairwise distinct in
	 * each of their attribute values.
	 * 
	 * @param benchmarkName
	 * @param composition
	 * @param minNum
	 *            strictly positive minimum number of samples
	 * @return
	 */
	public boolean kCompletelyDistinctSamplesAvailable(String benchmarkName, ComponentInstance composition,
			int minNum) {
		String identifier = Util.getComponentNamesOfComposition(composition);
		if (!this.performanceInstancesByIdentifier.containsKey(benchmarkName))
			return false;
		if (!this.performanceInstancesByIdentifier.get(benchmarkName).containsKey(identifier))
			return false;
		Instances instances = performanceInstancesByIdentifier.get(benchmarkName).get(identifier);
		if (instances.numInstances() == 0)
			return false;
		int count = 0;
		if (minNum == 1 && instances.numInstances() > 0)
			return true;
		for (int i = 0; i < instances.numInstances(); i++) {
			boolean distinctFromAll = true;
			for (int j = 0; j < i; j++) {
				Instance instance1 = instances.get(i);
				Instance instance2 = instances.get(j);
				for (int k = 0; k < instances.numAttributes() - 1; k++) {
					if ((instances.attribute(k).isNominal() || instances.attribute(k).isString())
							&& (instances.attribute(k).numValues() < minNum)) {
						continue;
					} else if (instances.attribute(k).getUpperNumericBound() <= instances.attribute(k)
							.getLowerNumericBound()) {
						continue;
					}
					if (instance1.value(k) == instance2.value(k)) {
						distinctFromAll = false;
					}
				}
			}
			if (distinctFromAll)
				count++;
			if (count >= minNum)
				return true;
		}
		return false;
	}

	public Instances getPerformanceSamples(String benchmarkName, ComponentInstance composition) {
		String identifier = Util.getComponentNamesOfComposition(composition);
		if (this.performanceInstancesByIdentifier.get(benchmarkName) != null)
			return this.performanceInstancesByIdentifier.get(benchmarkName).get(identifier);
		else
			return null;
	}

	public Instances getPerformanceSamplesForIndividualComponent(String benchmarkName, Component component) {
		if (this.performanceInstancesIndividualComponents.get(benchmarkName) != null) {
			if (this.performanceInstancesIndividualComponents.get(benchmarkName).get(component.getName()) != null) {
				return this.performanceInstancesIndividualComponents.get(benchmarkName).get(component.getName());
			}
		}
		return null;
	}

	public int getNumSamplesForComponent(String benchmarkName, Component component) {
		if (this.performanceInstancesIndividualComponents.get(benchmarkName) != null) {
			if (this.performanceInstancesIndividualComponents.get(benchmarkName).get(component.getName()) != null) {
				return this.performanceInstancesIndividualComponents.get(benchmarkName).get(component.getName()).size();
			}
		}
		return 0;
	}

	public List<Parameter> deriveAllParameters(List<Component> allComponents) {
		return allComponents.stream().flatMap(c -> c.getParameters().stream()).collect(Collectors.toList());
	}

	/**
	 * Trains a RQP on the current Composition-Performance-Data and returns it
	 * 
	 * @param benchmarkName
	 * @param composition
	 * @return the trained RQP
	 * @throws Exception
	 *             if something went wrong during the training phase
	 */
	public RangeQueryPredictor getTrainedRQPForComposition(String benchmarkName, ComponentInstance composition)
			throws Exception {

		if (composition == null)
			return null;

		Instances samples = getPerformanceSamples(benchmarkName, composition);

	//	System.out.println("-----\nAll samples are:\n" + samples + "\n--\nfor componentInstance " + composition);

		// we dont' have enough data to make a prediction
		// this shouldn't be a problem as we can use a random-completer instead
		if (samples == null)
			return null;
		Instances preprocessed = preprocessForRPQ(samples);

	//	System.out.println("--\nPreprocessed instances are:\n" + preprocessed);
		ExtendedRandomForest rqp = new ExtendedRandomForest();
		rqp.buildClassifier(preprocessed);

		return rqp;

	}

	/**
	 * Preprocesses precise data points using one-hot-encoding on the categorical
	 * attributes.
	 * 
	 * @param samples
	 *            the dataset that should be preprocessed
	 * @return a preprocessed dataset (that has one hot encoding)
	 * @throws Exception
	 */
	private static Instances preprocessForRPQ(Instances samples) throws Exception {
		// get an instances object that has one hot encododed attributes in the header
		Instances header = getHeaderInstanceForPreprocessedSamples(samples);
		for (Instance instance : samples) {
			Instance preprocessedInstance = new DenseInstance(header.numAttributes());
			for (int i = 0; i < samples.numAttributes(); i++) {
				Attribute attr = instance.attribute(i);
				if (attr.isNumeric()) {
					// don't encode anything here
					preprocessedInstance.setValue(computeIndexForAttributeInPP(samples, i), instance.value(i));
				}
				if (attr.isNominal()) {
					// one hot encoding!
					String value = instance.stringValue(i);
					// starting attribute index in the preprocessed instance
					// e.g. if we have the instance [1.0, a, 4.6] where the second attribute can
					// have the values {a, b, c} then, the starting index for the second attribute
					// is 1, whereas the starting index for the third attribute is 4.
					int startingIndex = computeIndexForAttributeInPP(samples, i);
					// actual one hot encoding
					Enumeration<Object> nominalValues = attr.enumerateValues();
					// iterate over the set {a, b, c}
					while (nominalValues.hasMoreElements()) {
						Object nextValue = nominalValues.nextElement();
						if (value.equals(nextValue)) {
							preprocessedInstance.setValue(startingIndex, 1.0);
						} else {
							preprocessedInstance.setValue(startingIndex, 0.0);
						}
						startingIndex++;
					}
				}
			}
			header.add(preprocessedInstance);
		}
		header.setClassIndex(header.numAttributes()-1);
		return header;
	}

	/**
	 * Computes a header instances object from the given instance. That is, it will
	 * create the correct header information for the one hot encoding.
	 * 
	 * @param instance
	 * @return the header with one hot attributes
	 */
	private static Instances getHeaderInstanceForPreprocessedSamples(Instances instance) {
		ArrayList<Attribute> pAttributes = new ArrayList<>();
		for (int i = 0; i < instance.numAttributes(); i++) {
			Attribute attr = instance.attribute(i);
			if (attr.isNominal()) {
				Enumeration<Object> attrEnum = attr.enumerateValues();
				while (attrEnum.hasMoreElements()) {
					Object possibleValue = attrEnum.nextElement();
					pAttributes.add(new Attribute("one_hot_attr_" + attr.name() + "_index_" + possibleValue, false));
				}
			}
			if (attr.isNumeric()) {
				pAttributes.add(attr);
			}
		}
		return new Instances(instance.relationName(), pAttributes, instance.size());
	}

	private static int computeIndexForAttributeInPP(Instances instances, int attrIndex) {
		int index = 0;
		for (int i = 0; i < attrIndex; i++) {
			Attribute attr = instances.attribute(i);
			if (attr.isNumeric()) {
				index++;
			} else {
				// assuming one-hot
				index += attr.numValues();
			}
		}
		return index;
	}

	/**
	 * Derives the number of attributes that are needed to convert a precise data
	 * point into a range query. That is,
	 * 
	 * every (precise) numeric attribute will yield 2 attributes in the range query
	 * (lower + upper bound)
	 * 
	 * every (precise) categorical attribute will yield n attributes in the range
	 * query (where n is the number of categorical features in the domain)
	 * 
	 * @param sample
	 * @return the number of attributes in a range query
	 */
	public int deriveNumAttributesForRangeQuery(List<Parameter> sample) {
		int sum = 0;
		for (Parameter param : sample) {
			if (param.isNumeric()) {
				sum += 2;
			} else if (param.isCategorical()) {
				int n = ((CategoricalParameterDomain) param.getDefaultDomain()).getValues().length;
				sum += n;
			}
		}
		return sum;
	}

	/**
	 * Derives the attribute index in a range query for the concrete parameter e.g.
	 * if the pipeline has 2 parameters A (numeric) and B (categorical with 3
	 * values). Then, the index of parameter B is 2, since a range-query would look
	 * as follows: [a_min, a_max, b_1, b_2, b_3] (where b is one hot encoded)
	 * 
	 * @param params
	 *            the sorted list of parameters
	 * @param indexedParameter
	 * @return the starting index of the parameter in a range query
	 */
	public int getStartIndexForParameterInRangeQuery(List<Parameter> params, Parameter indexedParameter) {
		int index = 0;
		for (Parameter param : params) {
			if (param.equals(indexedParameter)) {
				return index;
			}
			if (param.isNumeric()) {
				// lower + upper
				index += 2;
			} else if (param.isCategorical()) {
				// we need n attributes for one hot encoding
				int n = ((CategoricalParameterDomain) param.getDefaultDomain()).getValues().length;
				index += n;
			}
		}
		throw new IllegalStateException("The indexed parameter was not a part of the parameters!");
	}

	/**
	 * Inner helper class for managing parameter configurations easily.
	 * 
	 * @author jmhansel
	 *
	 */
	private class ParameterConfiguration {
		private final List<Pair<Parameter, String>> values;

		public ParameterConfiguration(ComponentInstance composition) {
			ArrayList<Pair<Parameter, String>> temp = new ArrayList<>();
			List<ComponentInstance> componentInstances = Util.getComponentInstancesOfComposition(composition);
			for (ComponentInstance compInst : componentInstances) {
				PartialOrderedSet<Parameter> parameters = compInst.getComponent().getParameters();
				for (Parameter parameter : parameters) {
					temp.add(Pair.of(parameter, compInst.getParameterValues().get(parameter.getName())));
				}
			}
			// Make the list immutable to avoid problems with hashCode
			values = Collections.unmodifiableList(temp);
		}

		@Override
		public int hashCode() {
			return values.hashCode();
		}

		public List<Pair<Parameter, String>> getValues() {
			return this.values;
		}
	}
}
