package com.example;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.util.*;
import java.io.IOException;
import java.util.StringTokenizer;
import java.util.regex.Pattern;

public class CityPartitioner extends Configured implements Tool{

    /**
     * Main function which calls the run method and passes the args using ToolRunner
     * @param args Two arguments input and output file paths
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new CityPartitioner(), args);
        System.exit(exitCode);
    }
    /**
     * Run method which schedules the Hadoop Job
     * @param args Arguments passed in main function
     */
    public int run(String[] args) throws Exception {

        if (args.length != 2) {
            System.err.printf("Usage: %s needs two arguments <input> <output> files\n",
                    getClass().getSimpleName());
            return -1;
        }

        //Initialize the Hadoop job and set the jar as well as the name of the Job
        Configuration conf = this.getConf();
        Job job = Job.getInstance(conf);
        job.setJarByClass(CityPartitioner.class);
        job.setJobName("City Partitioner");

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        //Set the MapClass and ReduceClass in the job
        job.setMapperClass(TokenizerMapper.class);
        job.setReducerClass(IntSumReducer.class);
        job.setPartitionerClass(MyPartitioner.class);
        job.setNumReduceTasks(9); //The output file will be split into 9 parts


        //Wait for the job to complete and print if the job was successful or not
        int returnValue = job.waitForCompletion(true) ? 0:1;

        if(job.isSuccessful()) {
            System.out.println("Job was successful");
        } else if(!job.isSuccessful()) {
            System.out.println("Job was not successful");
        }

        return returnValue;
    }


    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {




        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();



        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            String words = value.toString();
            context.write(new Text(words), one);





        }
    }

    public static class IntSumReducer extends Reducer<Text,IntWritable,Text,IntWritable> {

        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static class MyPartitioner extends Partitioner<Text, Writable>
    {
        //We are designating reducers based only on departing city; therefore, we don't need the entire key
        public Text getDepartureCity(Text key){
            String city = key.toString().replaceAll("[^a-zA-Z0-9,]", "");
            return new Text(city.substring(0, city.indexOf(",")));
        }
        //Send city to designated reducer. There may not be enough reducers available, so any given reducer can hold
        // multiple departure cities
        @Override
        public int getPartition(Text key, Writable value, int numPartitions)
        {
            return getDepartureCity(key).hashCode() % numPartitions; //Since there are 9 partitions, the first 9 unique
            // cities will have their own reducer
        }
        //@Override
        public void configure(JobConf conf) { }


    }

}
