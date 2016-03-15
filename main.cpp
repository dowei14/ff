#include <utils/ann-framework/ann.h>
#include <utils/ann-framework/backpropagation.h>
#include <utils/ann-framework/neuron.h>
#include <iostream>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

using namespace std;

#define RUNS 2
#define ITERATIONS 10
#define MINI_BATCH 100
#define EPOCHSPERBATCH 1

#define SAMPLES 41653
#define INPUTS 11
#define OUTPUTS 7
#define HIDDEN 10

std::string inTrain = "../data/in11.txt";
std::string outTrain = "../data/out7.txt";
std::string inTest = "../data/in11.txt";
std::string outTest = "../data/out7.txt";


double train_in[SAMPLES][INPUTS];
double train_out[SAMPLES][OUTPUTS];

double test_in[SAMPLES][INPUTS];
double test_out[SAMPLES][OUTPUTS];


class StateNN : public ANN
{
	public:
		StateNN();
		enum { nn_inputs = INPUTS, nn_hidden = HIDDEN, nn_outputs=OUTPUTS };		
};

void loadData();
void store(int run, int iteration, double error);
Backpropagation createTrainer(StateNN &ann);
TrainingPattern* getTrainingSample();

int main(int argc, char **argv) {
	loadData();

	for (int q=0;q<RUNS;q++){
		cout<<"run: "<<q<<endl;
		StateNN ann;
		//cout<<ann.dumpWeights()<<endl;
		ann.updateTopologicalSort(); // create a topolocial sorting of our network. This is required for the backpropagation algorithm as "Full Batch mode"
		for (int a=0;a<ITERATIONS;a++){
			// create backpropagation object
			Backpropagation trainer;
			trainer = createTrainer(ann);
			// create mini_batch
			for (int i=0;i<MINI_BATCH;i++){
				trainer.addTrainingPattern(getTrainingSample());
			}
			// train
			trainer.learn(EPOCHSPERBATCH);

			// test
			long errors = 0;
			for (long n=0;n<SAMPLES;n++){		
				for (int i=0;i<INPUTS;i++){
					ann.setInput(i,test_in[n][i]);
				}
				ann.feedForwardStep();
				double max = -9999;
				int maxID = 0;
				for (int i=0;i<OUTPUTS;i++){
					double val = ann.getOutput(i);
					if (val > max) {
						max = val;
						maxID = i;
					}
				}
				bool error = false;
				for (int i=0;i<OUTPUTS;i++){
					double val;
					if (i == maxID) val = 1.0;
					else val = 0.0;
					if (val != test_out[n][i]) error = true;
				}
				if (error) errors++;
			}
			cout<<(double)errors/SAMPLES<<endl;
			store(q,(a+1)*10,(double)errors/SAMPLES);
			//cout<<ann.dumpWeights()<<endl;
		}
	}
}

void store(int run, int iteration, double error){
	ofstream results;
	results.open("results.txt", ios::app);
	results.precision(5);
	results<<fixed;
	results << run << " " << iteration << " " << error <<"\n";
	results.close();
}


void loadData(){
	ifstream data_inputs(inTrain.c_str()); //opening an input stream for file test.txt
	ifstream data_outputs(outTrain.c_str()); //opening an input stream for file test.txt
	std::string line_out;

	long trainer_counter = 0;
	for(std::string line_in; std::getline(data_inputs, line_in); )   //read stream line by line
	{		
		// add inputs
		std::istringstream in(line_in);      //make a stream for the line itself
		//cout<<"Inputs: ";
		for (int i=0;i<INPUTS;i++){
			in >> train_in[trainer_counter][i];                  //and read the first whitespace-separated token			
			//cout<<train_in[trainer_counter][i]<<" ";
		}

		// add outputs
		std::getline(data_outputs, line_out);
		std::istringstream out(line_out);
		//cout<<" Outputs: ";
		for (int o=0;o<OUTPUTS;o++){
			out >> train_out[trainer_counter][o];                  //and read the first whitespace-separated token		
			//cout<<train_out[trainer_counter][o]<<" ";
		}
		
		//cout<<endl;
		trainer_counter++;
	}

	ifstream data_inputs2(inTest.c_str()); //opening an input stream for file test.txt
	ifstream data_outputs2(outTest.c_str()); //opening an input stream for file test.txt
	long test_counter = 0;
	for(std::string line_in; std::getline(data_inputs2, line_in); )   //read stream line by line
	{
		
		// add inputs
		std::istringstream in(line_in);      //make a stream for the line itself
		//cout<<"Inputs: ";
		for (int i=0;i<INPUTS;i++){
			in >> test_in[test_counter][i];                  //and read the first whitespace-separated token
			
		}
		
		std::getline(data_outputs2, line_out);
		std::istringstream out(line_out);

		for (int o=0; o<OUTPUTS;o++){
			out >> test_out[test_counter][o];
		}
		test_counter++;
	}
	cout<<"Train"<<endl;
	for (int i=0;i<INPUTS;i++) std::cout<<train_in[0][i]<<" ";
	cout<<endl;
	for (int i=0;i<OUTPUTS;i++) std::cout<<train_out[0][i]<<" ";
	cout<<endl;
	cout<<"Test"<<endl;
	for (int i=0;i<INPUTS;i++) std::cout<<test_in[0][i]<<" ";
	cout<<endl;
	for (int i=0;i<OUTPUTS;i++) std::cout<<test_out[0][i]<<" ";
	cout<<endl;	
}

StateNN::StateNN() {

	//set transfer function
//	setDefaultTransferFunction(ANN::tanhFunction());
	setDefaultTransferFunction(ANN::logisticFunction());

	setNeuronNumber(nn_inputs+nn_hidden+nn_outputs); // total number of neurons

	// create random number generator between -1 and 1
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_real_distribution<> distr(-1, 1); // define the range

	// initialize weights
	for (int h=nn_inputs; h<(nn_inputs+nn_hidden);h++){
		for (int i=0; i<nn_inputs;i++){
			w(h,i,distr(eng));
		}
	}
	for (int o=nn_inputs+nn_hidden; o<(nn_inputs+nn_hidden+nn_outputs);o++){
		for (int h=nn_inputs; h<(nn_inputs+nn_hidden);h++){
			w(o,h,distr(eng));
		}
	}
}

Backpropagation createTrainer(StateNN &ann){
	Backpropagation trainer;
	trainer.setNeuralNetwork(&ann);
	for (int i=0; i<ann.nn_inputs;i++){
		trainer.defineInputNeuron(i, ann.getNeuron(i));
	}
	int output_counter = 0;
	for (int o=ann.nn_inputs+ann.nn_hidden; o<(ann.nn_inputs+ann.nn_hidden+ann.nn_outputs);o++){
		trainer.defineOutputNeuron(output_counter, ann.getNeuron(o));
		output_counter++;
	}	
	trainer.includeAllSynapses();
	trainer.includeAllNeuronBiases();
	trainer.setLearningRate(0.01);
	return trainer;
}

TrainingPattern* getTrainingSample(){
	TrainingPattern* p = new TrainingPattern;
	std::random_device rd; // obtain a random number from hardware
 	std::mt19937 eng(rd()); // seed the generator
	std::uniform_real_distribution<> distr(0, (SAMPLES-1)); // define the range
	int run_id =(int)distr(eng);
	for (int n=0;n<INPUTS;n++){
		p->inputs[n] = train_in[run_id][n];
	}
	for (int n=0;n<OUTPUTS;n++){
		p->outputs[n] = train_out[run_id][n];
	}
	return p;
}

