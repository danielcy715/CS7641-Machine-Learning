/*
 *  How to use WEKA API in Java
 *  Copyright (C) 2014
 *  @author Dr Noureddin M. Sadawi (noureddin.sadawi@gmail.com)
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it as you wish ...
 *  I ask you only, as a professional courtesy, to cite my name, web page
 *  and my YouTube Channel!
 *
 */
//import required classes

import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class LR_ICAWithFilter {
    public static void main(String args[]) throws Exception{
        //load dataset
        DataSource source = new DataSource("letter_ica.arff");
        Instances dataset = source.getDataSet();
        //set class index to the last attribute
        dataset.setClassIndex(dataset.numAttributes()-1);

        Random rand = new Random(10);

        //the base classifier
        J48 tree = new J48();
        //the filter
        List<List<String>> opts = new ArrayList<List<String>>();
        for (int i = 1; i < 17; i++){
            List<String> opt = new ArrayList<String>();
            opt.add("-R");
            String str = "";
            for (int j = i + 1; j <= 16; j++){
                str = str + Integer.toString(j) + ",";
            }
            if (str.length() > 0) {
                str = str.substring(0, str.length() - 1);
            }
            opt.add(str);
            opts.add(opt);

        }
        //remove.setAttributeIndices("1");
        String[] opts0 = new String[]{ "-R", "2,3,4,5,6,7,8,9,10"};
        String[] opts1 = new String[]{ "-R", "3,4,5,6,7,8,9,10"};
        String[] opts2 = new String[]{ "-R", "4,5,6,7,8,9,10"};
        String[] opts3 = new String[]{ "-R", "5,6,7,8,9,10"};
        String[] opts4 = new String[]{ "-R", "6,7,8,9,10"};
        String[] opts5 = new String[]{ "-R", "7,8,9,10"};
        String[] opts6 = new String[]{ "-R", "8,9,10"};
        String[] opts7 = new String[]{ "-R", "9,10"};
        String[] opts8 = new String[]{ "-R", "10"};
        String[] opts9 = new String[]{ "-R", ""};
        //String[][] opts = new String[][]{opts0, opts1,opts2,opts3,opts4,opts5,opts6,opts7,opts8,opts9};
        //set the filter options
        List<String> accuracy = new ArrayList<String>();
        for (int i = 0; i < 16; i++) {

            Remove remove = new Remove();
            List<String> opt = opts.get(i);
            String [] option = opt.toArray(new String[opt.size()]);
            remove.setOptions(option);

            //Create the FilteredClassifier object
            FilteredClassifier fc = new FilteredClassifier();
            //specify filter
            fc.setFilter(remove);
            //specify base classifier
            fc.setClassifier(tree);
            //Build the meta-classifier
            fc.buildClassifier(dataset);

            Evaluation eval = new Evaluation(dataset);
            eval.crossValidateModel(fc, dataset, 10, rand);
            System.out.println("options " + i + " Correct % = " + eval.pctCorrect());
            accuracy.add(Double.toString(eval.pctCorrect()));
        }
        String collect = accuracy.stream().collect(Collectors.joining(","));
        System.out.println(collect);

        FileWriter writer = new FileWriter("letter_ica_accuracy.csv", true);
        writer.write(collect);
        writer.write("\n");
        writer.close();
        //System.out.println(tree.graph());
    }

}