import org.apache.commons.lang.StringUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

/**
 * Created by colntrev on 2/24/18.
 */
public class SparkKMeans {

    private static class ParsePoint implements Function<String, Vector> {

        @Override
        public Vector call(String v1) throws Exception {
            String[] tokens = StringUtils.split(v1, ",");
            double[] point = new double[tokens.length];
            for(int i = 0; i < tokens.length; i++){
                point[i] = Double.parseDouble(tokens[i]);
            }
            return Vectors.dense(point);
        }
    }
    public static void main(String[] args){
        if(args.length < 3){
            System.err.println("Usage: SparkKMeans <input file> <k> <iterations> <runs..optional>");
            System.exit(-1);
        }
        String inputFile = args[0];
        int k = Integer.parseInt(args[1]);
        int iterations = Integer.parseInt(args[2]);
        int runs = args.length >= 4? Integer.parseInt(args[3]) : 1;
        SparkConf conf = new SparkConf().setAppName("KMeans");
        JavaSparkContext context = new JavaSparkContext(conf);
        long startTime = System.currentTimeMillis();
        JavaRDD<String> lines = context.textFile(inputFile);
        JavaRDD<Vector> points = lines.map(new ParsePoint());

        KMeansModel model = KMeans.train(points.rdd(), k, iterations, runs, KMeans.RANDOM());

        System.out.println("Cluster Centers");
        for(Vector center : model.clusterCenters()){
            System.out.println(" " + center);
        }

        double cost = model.computeCost(points.rdd());
        System.out.println("Cost: " + cost);
        long endTime = System.currentTimeMillis();
        context.close();
        System.out.println("Elapsed Time: " + (endTime - startTime));
    }
}
