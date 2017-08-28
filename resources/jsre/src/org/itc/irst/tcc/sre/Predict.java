/*
 * Copyright 2005 FBK-irst (http://www.fbk.eu)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.itc.irst.tcc.sre;

import java.util.*;
import java.io.*;

import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
import org.itc.irst.tcc.sre.kernel.expl.Mapping;
import org.itc.irst.tcc.sre.kernel.expl.MappingFactory;
import org.itc.irst.tcc.sre.kernel.KernelNotFoundException;
import org.itc.irst.tcc.sre.util.svm_train;
import org.itc.irst.tcc.sre.util.UnZipModel;
import org.itc.irst.tcc.sre.util.FeatureIndex;
import org.itc.irst.tcc.sre.util.svm_predict;
import org.itc.irst.tcc.sre.util.Evaluator;
import org.itc.irst.tcc.sre.data.*;
import libsvm.*;

/**
 * TO DO
 *
 * @author 	Claudio Giuliano
 * @version %I%, %G%
 * @since		1.0
 */
public class Predict
{
	/**
	 * Define a static logger variable so that it references the
	 * Logger instance named <code>Predict</code>.
	 */
	static Logger logger = Logger.getLogger(Predict.class.getName()); 

	//
	private File inputFile;
	
	//
	private File modelFile;

	//
	private File outputFile;

	//
	private String kernel;
	
	//
	private int relationType;

	//
	private Properties parameter;

	//
	public Predict(String input, String model, String output)
	{
		this(new File(input), new File(model), new File(output));
	} // end constructor

	
	//
	public Predict(File inputFile, File modelFile, File outputFile)
	{
		this.inputFile = inputFile;
		this.modelFile = modelFile;
		this.outputFile = outputFile;
		parameter = new Properties();
	} // end constructor
	
	// read parameters
	private void readParameters(UnZipModel model) throws IOException
	{
		logger.info("read parameters");

		// get the param model
		File paramFile = model.get("param");
		parameter.load(new FileInputStream(paramFile));
	} // end readParameters

	// read the data set
	private ExampleSet readDataSet(File in) throws IOException
	{
		logger.info("read the example set");
		
		// 
		logger.info("read data set");
		ExampleSet inputSet = new SentenceSetCopy();
		inputSet.read(new BufferedReader(new FileReader(in)));

		return inputSet;
	}	// end readDataSet

	//
	private FeatureIndex[] readFeatureIndex(int subspaceCount, UnZipModel model) throws IOException
	{
		logger.info("read feature index: " + subspaceCount);
		
		FeatureIndex[] index = new FeatureIndex[subspaceCount];
		logger.info("index length " + index.length);
		for (int i=0;i<subspaceCount;i++)
		{
			index[i] = new FeatureIndex(true, 1);
			File dicFile = model.get("dic" + i);
			logger.info(i + " read " + dicFile);
			BufferedReader br = new BufferedReader(new FileReader(dicFile));
			index[i].read(br);
			br.close();

			logger.debug("dic" + i + ", " + dicFile + ", " + index[i].size());
		} // end for
		
		return index;
	} // end readFeatureIndexes

	// save the embedded test set
	private File saveExampleSet(ExampleSet outputSet) throws IOException
	{
		logger.info("save the embedded test set");
		
		// 
		File tmp = File.createTempFile("test", null);
		tmp.deleteOnExit();
		//File tmp = new File("tmp/test");
		logger.debug(tmp.getName() + " written--------------------------");
		
		BufferedWriter out = new BufferedWriter(new FileWriter(tmp));
		outputSet.write(out);
		out.close();
		
		return tmp;
	} // end saveExampleSet

	//
	public void run() throws Exception
	{
		logger.info("predict relations");
		
		// open zip archive
		UnZipModel model = new UnZipModel(modelFile);
		
		// get the param model
		readParameters(model);
		
		// read data set
		ExampleSet inputSet = readDataSet(inputFile);

		// find argument types
		ArgumentSet.getInstance().init(inputSet);
		
		// create the mapping factory
		MappingFactory mappingFactory = MappingFactory.getMappingFactory();
		Mapping mapping = mappingFactory.getInstance(parameter.getProperty("kernel-type"));
		
		// set the parameters
		mapping.setParameters(parameter);
		
		// get the number of subspaces
		int subspaceCount = mapping.subspaceCount();

		// read the index
		FeatureIndex[] index = readFeatureIndex(subspaceCount, model);
		
		// embed the input data into a feature space
		logger.info("embed the test set");
		ExampleSet outputSet = mapping.map(inputSet, index);
		logger.debug("embedded test set size: " + outputSet.size());

		// save the test set
		File svmTestFile = saveExampleSet(outputSet);
		logger.info("run svm predict");
				
		//<save memory?>
		System.err.println("Saving memory");
		index = null;
		outputSet = null;
		inputSet = null;
	        System.gc(); 

		// get the svm model
		File svmModelFile = model.get("model");
		
		// predict
		logger.info("run svm predict");
		svm_predict.run(svmTestFile.getAbsolutePath(), svmModelFile.getAbsolutePath(), outputFile.getAbsolutePath());

	} // end main

	//
	public static void main(String args[]) throws Exception
	{
		String logConfig = System.getProperty("log-config");
		if (logConfig == null)
			logConfig = "log-config.txt";
			
		PropertyConfigurator.configure(logConfig);
		
		int parm = 3;
		
		if (args.length != parm)
		{
			System.out.println(getHelp());
			System.exit(-1);
		}

		File inputFile = new File(args[args.length - 3]);
		File modelFile = new File(args[args.length - 2]);
		File outputFile = new File(args[args.length - 1]);
		
		Predict predict = new Predict(inputFile, modelFile, outputFile);
		predict.run();

		//logger.info("evaluate predictions");
		//Excluded next line by Philippe
		// Evaluator eval = new Evaluator(inputFile, outputFile);
		// logger.info("micro\ttp\tfp\tfn\ttotal\tprec\trecall\tF1");
		// logger.info(eval);
		
	} // end main

	/**
	 * Returns a command-line help.
	 *
	 * return a command-line help.
	 */
	private static String getHelp()
	{
		StringBuffer sb = new StringBuffer();

		// SRE
		sb.append("\njSRE: Simple Relation Extraction V1.10\t 30.08.06\n");
		sb.append("developed by Claudio Giuliano (giuliano@itc.it)\n\n");

		// License
		sb.append("Copyright 2005 FBK-irst (http://www.fbk.eu)\n");
		sb.append("\n");
		sb.append("Licensed under the Apache License, Version 2.0 (the \"License\");\n");
		sb.append("you may not use this file except in compliance with the License.\n");
		sb.append("You may obtain a copy of the License at\n");
		sb.append("\n");
		sb.append("    http://www.apache.org/licenses/LICENSE-2.0\n");
		sb.append("\n");
		sb.append("Unless required by applicable law or agreed to in writing, software\n");
		sb.append("distributed under the License is distributed on an \"AS IS\" BASIS,\n");
		sb.append("WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n");
		sb.append("See the License for the specific language governing permissions and\n");
		sb.append("limitations under the License.\n\n");
		
		// Usage
		sb.append("Usage: java org.itc.irst.tcc.sre.Predict [options] example-file model-file output-file\n\n");

		// Arguments
		sb.append("Arguments:\n");
		sb.append("\ttest-file\t-> file with test data (SRE format)\n");
		sb.append("\tmodel-file\t-> file from which to load the learned model\n");
		sb.append("\toutput-file\t-> file in which to store resulting output\n");
		
		sb.append("Options:\n");
		sb.append("\t-h\t\t-> this help\n");
		
		return sb.toString();
	} // end getHelp

} // end class Predict
