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
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class travellingsales_ga {
    /** The n value */
    private static final int N = 50;

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



        double start, end, time;
        //int[] iters = {10,100,500,1000,2500,5000};
        int[] iters = {5000};
        int testRuns = 10;

        int[] population = {10,20,50,100,200,500};
        int[] mate = {5,10,25,50,150,350};
        int[] mute = {2,5,10,10,20,50};


        //int[] population = {200};
        //int[] mate = {150};
        //int[] mute = {20};


        double sum_ga = 0;


        double time_ga = 0;


        for (int i = 0; i < population.length; i++) {

            sum_ga = 0;
            time_ga = 0;



            for (int j = 0; j < testRuns; j++) {

                Random random = new Random();
                // create the random points
                double[][] points = new double[N][2];
                for (int q = 0; q < points.length; q++) {
                    points[q][0] = random.nextDouble();
                    points[q][1] = random.nextDouble();
                }
                // for rhc, sa, and ga we use a p
                // ermutation based encoding
                TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
                Distribution odd = new DiscretePermutationDistribution(N);
                NeighborFunction nf = new SwapNeighbor();
                MutationFunction mf = new SwapMutation();
                CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
                HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
                GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

                start = System.nanoTime();
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(population[i], mate[i], mute[i], gap);
                FixedIterationTrainer fit = new FixedIterationTrainer(ga, 5000);
                fit.train();
                end = System.nanoTime();
                time = end - start;
                time /= Math.pow(10, 9);
                sum_ga += ef.value(ga.getOptimal());
                time_ga += time;
                //System.out.println("ga: " + ef.value(ga.getOptimal()));
                //System.out.println(time);


            }


            double average_ga = sum_ga / testRuns;

            double averagetime_ga = time_ga / testRuns;


            System.out.println("##############");
            System.out.println("ga average is " + average_ga + "," + population[i] + "," + mate[i] + "," + mute[i]+ ", time average is " + averagetime_ga);

            String final_result = "";
            final_result = "ga" + "," + population[i] + "," + mate[i] + "," + mute[i] + "," + Double.toString(average_ga) + "," + Double.toString(averagetime_ga);

            write_output_to_file("Optimization_Results", "travelingsalesman_ga_results.csv", final_result, true);
        }

    }

}
