package opt.ycai87;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test of the knap sack problem
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class Knapsack {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 40;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum volume for a single element */
    private static final double MAX_VOLUME = 50;
    /** The volume of the knapsack */
    private static final double KNAPSACK_VOLUME =
            MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4;



    public static void write_output_to_file(String output_dir, String file_name, String results, boolean final_result) {
        try {
            if (final_result) {
                String augmented_output_dir = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date());
                String full_path = augmented_output_dir + "/" + file_name;
                Path p = Paths.get(full_path);
                if (Files.notExists(p)) {
                    Files.createDirectories(p.getParent());
                }
                PrintWriter pwtr = new PrintWriter(new BufferedWriter(new FileWriter(full_path, true)));
                synchronized (pwtr) {
                    pwtr.println(results);
                    pwtr.close();
                }
            }
            else {
                String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) + "/" + file_name;
                Path p = Paths.get(full_path);
                Files.createDirectories(p.getParent());
                Files.write(p, results.getBytes());
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }
    public static void main(String[] args) {


        /*Editted for Clock Time*/

        double start, end, time;
        int[] iters = {10,100,500,1000,2500,5000};
        //int[] iters = {10};
        int testRuns = 10;


        for (int iter : iters) {
            int sum_rhc = 0;
            int sum_sa = 0;
            int sum_ga = 0;
            int sum_mimic = 0;

            double time_rhc = 0;
            double time_sa = 0;
            double time_ga = 0;
            double time_mimic = 0;
            for (int j = 0; j < testRuns; j++) {

                int[] copies = new int[NUM_ITEMS];
                Arrays.fill(copies, COPIES_EACH);
                double[] weights = new double[NUM_ITEMS];
                double[] volumes = new double[NUM_ITEMS];
                for (int i = 0; i < NUM_ITEMS; i++) {
                    weights[i] = random.nextDouble() * MAX_WEIGHT;
                    volumes[i] = random.nextDouble() * MAX_VOLUME;
                }
                int[] ranges = new int[NUM_ITEMS];
                Arrays.fill(ranges, COPIES_EACH + 1);
                EvaluationFunction ef = new KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies);
                Distribution odd = new DiscreteUniformDistribution(ranges);
                NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
                MutationFunction mf = new DiscreteChangeOneMutation(ranges);
                CrossoverFunction cf = new UniformCrossOver();
                Distribution df = new DiscreteDependencyTree(.1, ranges);
                HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
                GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
                ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
                //System.out.println("this is test run # " + j);
                start = System.nanoTime();
                RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iter);
                fit.train();
                end = System.nanoTime();
                time = end - start;
                time /= Math.pow(10, 9);
                sum_rhc += ef.value(rhc.getOptimal());
                time_rhc += time;
                //System.out.println("rhc: " + ef.value(rhc.getOptimal()));
                //System.out.println(time);

                start = System.nanoTime();
                SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
                fit = new FixedIterationTrainer(sa, iter);
                fit.train();
                end = System.nanoTime();
                time = end - start;
                time /= Math.pow(10, 9);
                sum_sa += ef.value(sa.getOptimal());
                time_sa += time;
                //System.out.println("sa: " + ef.value(sa.getOptimal()));
                //System.out.println(time);

                start = System.nanoTime();
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
                fit = new FixedIterationTrainer(ga, iter);
                fit.train();
                end = System.nanoTime();
                time = end - start;
                time /= Math.pow(10, 9);
                sum_ga += ef.value(ga.getOptimal());
                time_ga += time;
                //System.out.println("ga: " + ef.value(ga.getOptimal()));
                //System.out.println(time);

                start = System.nanoTime();
                MIMIC mimic = new MIMIC(200, 20, pop);
                fit = new FixedIterationTrainer(mimic, iter);
                fit.train();
                end = System.nanoTime();
                time = end - start;
                time /= Math.pow(10, 9);
                sum_mimic += ef.value(mimic.getOptimal());
                time_mimic += time;
                //System.out.println("Mimic: " + ef.value(mimic.getOptimal()));
                //System.out.println(time);

            }


            int average_rhc = sum_rhc / testRuns;
            int average_sa = sum_sa / testRuns;
            int average_ga = sum_ga / testRuns;
            int average_mimic = sum_mimic / testRuns;

            double averagetime_rhc = time_rhc / testRuns;
            double averagetime_sa = time_sa / testRuns;
            double averagetime_ga = time_ga / testRuns;
            double averagetime_mimic = time_mimic / testRuns;

            System.out.println("##############");
            System.out.println("this is iteration " + iter);
            System.out.println("rhc average is " + average_rhc + ", time average is " + averagetime_rhc);
            System.out.println("sa average is " + average_sa + ", time average is " + averagetime_sa);
            System.out.println("ga average is " + average_ga + ", time average is " + averagetime_ga);
            System.out.println("mimic average is " + average_mimic + ", time average is " + averagetime_mimic);

            String final_result = "";
            final_result = "rhc" + "," + iter + "," + Integer.toString(average_rhc) + "," + Double.toString(averagetime_rhc) + "," +
                            "sa" + "," + iter + "," + Integer.toString(average_sa) + "," + Double.toString(averagetime_sa) + "," +
                            "ga" + "," + iter + "," + Integer.toString(average_ga) + "," + Double.toString(averagetime_ga) + "," +
                            "mimic" + "," + iter + "," + Integer.toString(average_mimic) + "," + Double.toString(averagetime_mimic);

            write_output_to_file("Optimization_Results", "knapsack_results.csv", final_result, true);
        }

        int [] samples = {10, 20, 40, 80, 160,200,200,200,200, 200};
        int [] tokeep = {5,10,20,40,80,100,20,40,80,160,180};

        for (int i = 0; i < samples.length; i++) {

            int sum_mimic = 0;


            double time_mimic = 0;
            for (int j = 0; j < testRuns; j++) {

                int[] copies = new int[NUM_ITEMS];
                Arrays.fill(copies, COPIES_EACH);
                double[] weights = new double[NUM_ITEMS];
                double[] volumes = new double[NUM_ITEMS];
                for (int q = 0; q < NUM_ITEMS; q++) {
                    weights[q] = random.nextDouble() * MAX_WEIGHT;
                    volumes[q] = random.nextDouble() * MAX_VOLUME;
                }
                int[] ranges = new int[NUM_ITEMS];
                Arrays.fill(ranges, COPIES_EACH + 1);
                EvaluationFunction ef = new KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies);
                Distribution odd = new DiscreteUniformDistribution(ranges);
                NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
                MutationFunction mf = new DiscreteChangeOneMutation(ranges);
                CrossoverFunction cf = new UniformCrossOver();
                Distribution df = new DiscreteDependencyTree(.1, ranges);
                HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
                GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
                ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
                //System.out.println("this is test run # " + j);
                FixedIterationTrainer fit;

                start = System.nanoTime();
                MIMIC mimic = new MIMIC(samples[i], tokeep[i], pop);
                fit = new FixedIterationTrainer(mimic, 5000);
                fit.train();
                end = System.nanoTime();
                time = end - start;
                time /= Math.pow(10, 9);
                sum_mimic += ef.value(mimic.getOptimal());
                time_mimic += time;
                //System.out.println("Mimic: " + ef.value(mimic.getOptimal()));
                //System.out.println(time);

            }



            int average_mimic = sum_mimic / testRuns;


            double averagetime_mimic = time_mimic / testRuns;

            System.out.println("##############");
            System.out.println("this is sample " + samples[i]);

            System.out.println("mimic average is " + average_mimic + ", time average is " + averagetime_mimic);

            String final_result = "";
            final_result =
                    "mimic" + "," + Integer.toString(samples[i]) + "," + Integer.toString(tokeep[i])+ "," + Integer.toString(average_mimic) + "," + Double.toString(averagetime_mimic);

            write_output_to_file("Optimization_Results", "knapsack_mimic_results.csv", final_result, true);
        }
    }

}