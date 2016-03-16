#include <utils/ann-framework/ann.h>
#include <utils/ann-framework/backpropagation.h>
#include <utils/ann-framework/neuron.h>
#include <iostream>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

using namespace std;

// Training Setup
#define RUNS 1
#define ITERATIONS 100000
#define MINI_BATCH 100
#define EPOCHSPERBATCH 1
#define LEARNINGRATE 0.01

// NN Setup
#define INPUTS 18
#define OUTPUTS 7
#define HIDDEN 50
#define TRANSFER 1 // 1 = tanh, 2 = log

// Data Setup
/**/
#define BALANCETRIGGER 1

#define SAMPLES0 41524
#define SAMPLES1 129
std::string inTrain0 = "../data/T/0/in18.txt";
std::string outTrain0 = "../data/T/0/outT.txt";
std::string inTrain1 = "../data/T/1/in18.txt";
std::string outTrain1 = "../data/T/1/outT.txt";

double train_in0[SAMPLES0][INPUTS];
double train_out0[SAMPLES0][OUTPUTS];
double train_in1[SAMPLES1][INPUTS];
double train_out1[SAMPLES1][OUTPUTS];

#define SAMPLES 41653
std::string inTrain = "../data/in18.txt";
std::string outTrain = "../data/out7.txt";
double train_in[SAMPLES][INPUTS];
double train_out[SAMPLES][OUTPUTS];

/*#define SAMPLESTEST 41653
std::string inTest = "../data/in11.txt";
std::string outTest = "../data/outT.txt";
*/

#define SAMPLESTEST 129
std::string inTest = "../data/T/1/in18.txt";
std::string outTest = "../data/T/1/outT.txt";
double test_in[SAMPLESTEST][INPUTS];
double test_out[SAMPLESTEST][OUTPUTS];

double learn_out[SAMPLESTEST][OUTPUTS];
double learn_class_out[SAMPLESTEST][OUTPUTS];


class StateNN : public ANN
{
	public:
		StateNN();
		enum { nn_inputs = INPUTS, nn_hidden = HIDDEN, nn_outputs=OUTPUTS };		
};

// functions for standard training
void loadData();
TrainingPattern* getTrainingSample();
// functions for balanced training
void loadDataBalanceTrigger();
TrainingPattern* getTrainingSampleBlanced(bool positive);

// other stuff
void storeError(int run, int iteration, double mse, double classificationError);
void storeOutput();
Backpropagation createTrainer(StateNN &ann);




int main(int argc, char **argv) {

	if (BALANCETRIGGER == 1) loadDataBalanceTrigger();
	else loadData();

	for (int q=0;q<RUNS;q++){
		cout<<"run: "<<q<<endl;
		StateNN ann;
		//cout<<ann.dumpWeights()<<endl;
		// create a topolocial sorting of our network. This is required for the backpropagation algorithm as "Full Batch mode"
		ann.updateTopologicalSort();
		for (int a=0;a<ITERATIONS;a++){
			// create backpropagation object
			Backpropagation trainer;
			trainer = createTrainer(ann);
			// create mini_batch
			for (int i=0;i<MINI_BATCH;i++){
				if (BALANCETRIGGER == 0) trainer.addTrainingPattern(getTrainingSample());
				else {
					trainer.addTrainingPattern(getTrainingSampleBlanced(true));
					trainer.addTrainingPattern(getTrainingSampleBlanced(false));
				}
			}
			// train
			trainer.learn(EPOCHSPERBATCH);
if (a % 100 ==0) {
			// test
			long errors = 0;
			double mse = 0.0;
			for (long n=0;n<SAMPLESTEST;n++){		
				for (int i=0;i<INPUTS;i++){
					ann.setInput(i,test_in[n][i]);
				}
				ann.feedForwardStep();
				double max = -9999;
				int maxID = 0;
				for (int i=0;i<OUTPUTS;i++){
					double val = ann.getOutput(i);
					
					// add MSE
					mse += pow((test_out[n][i] - val), 2.0);										
					// find maximum value
					if (val > max) {
						max = val;
						maxID = i;
					}
					// store output
					learn_out[n][i] = ann.getOutput(i);
				}
				bool error = false;
				for (int i=0;i<OUTPUTS;i++){
					double val;
					if (i == maxID) val = 1.0;
					else val = 0.0;
					learn_class_out[n][i] = val;
					if (val != test_out[n][i]) error = true;
				}
				if (error) errors++;
			}
			double averageError = (double)mse/SAMPLESTEST;
			double classAverageError = (double)errors/SAMPLESTEST;
			if (a % 10 ==0) {
				std::cout<<"Iteration: "<<a<<" MSE: " <<averageError<< " CE:"<< classAverageError<<endl;
			}
			storeError(q,(a+1)*10,averageError, classAverageError);
			//cout<<ann.dumpWeights()<<endl;
}
		}
		//cout<<ann.dumpWeights()<<endl;
	}
	
	storeOutput();
}

void storeError(int run, int iteration, double mse, double classificationError){
	ofstream results;
	results.open("results.txt", ios::app);
	results.precision(5);
	results<<fixed;
	results << run << " " << iteration << " " << mse << " " << classificationError <<"\n";
	results.close();
}

void storeOutput(){
	ofstream output;
	output.open("outputs.txt", ios::out);
	output.precision(5);
	output<<fixed;
	ofstream outputClass;
	outputClass.open("outputs_class.txt", ios::out);
	outputClass.precision(0);
	outputClass<<fixed;
	for (long n=0;n<SAMPLESTEST;n++){		
		for (int i=0;i<OUTPUTS;i++){
			output << learn_out[n][i];
			outputClass << learn_class_out[n][i];
			if (i<(OUTPUTS-1)){
				output << " ";
				outputClass << " ";
			}
		}
		output <<"\n";
		outputClass << "\n";
	}
	output.close();
	outputClass.close();
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


void loadDataBalanceTrigger(){
	// negative samples
	ifstream data_inputs0(inTrain0.c_str()); //opening an input stream for file test.txt
	ifstream data_outputs0(outTrain0.c_str()); //opening an input stream for file test.txt
	std::string line_out0;

	long trainer_counter0 = 0;
	for(std::string line_in0; std::getline(data_inputs0, line_in0); )   //read stream line by line
	{		
		// add inputs
		std::istringstream in(line_in0);      //make a stream for the line itself
		//cout<<"Inputs: ";
		for (int i=0;i<INPUTS;i++){
			in >> train_in0[trainer_counter0][i];                  //and read the first whitespace-separated token			
			//cout<<train_in[trainer_counter][i]<<" ";
		}

		// add outputs
		std::getline(data_outputs0, line_out0);
		std::istringstream out(line_out0);
		//cout<<" Outputs: ";
		for (int o=0;o<OUTPUTS;o++){
			out >> train_out0[trainer_counter0][o];                  //and read the first whitespace-separated token		
			//cout<<train_out[trainer_counter][o]<<" ";
		}
		
		//cout<<endl;
		trainer_counter0++;
	}

	//positive samples
	ifstream data_inputs1(inTrain1.c_str()); //opening an input stream for file test.txt
	ifstream data_outputs1(outTrain1.c_str()); //opening an input stream for file test.txt
	std::string line_out1;

	long trainer_counter1 = 0;
	for(std::string line_in1; std::getline(data_inputs1, line_in1); )   //read stream line by line
	{		
		// add inputs
		std::istringstream in(line_in1);      //make a stream for the line itself
		//cout<<"Inputs: ";
		for (int i=0;i<INPUTS;i++){
			in >> train_in1[trainer_counter1][i];                  //and read the first whitespace-separated token			
			//cout<<train_in[trainer_counter][i]<<" ";
		}

		// add outputs
		std::getline(data_outputs1, line_out1);
		std::istringstream out(line_out1);
		//cout<<" Outputs: ";
		for (int o=0;o<OUTPUTS;o++){
			out >> train_out1[trainer_counter1][o];                  //and read the first whitespace-separated token		
			//cout<<train_out[trainer_counter][o]<<" ";
		}
		
		//cout<<endl;
		trainer_counter1++;
	}
	std::string line_out2;
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
		
		std::getline(data_outputs2, line_out2);
		std::istringstream out(line_out2);

		for (int o=0; o<OUTPUTS;o++){
			out >> test_out[test_counter][o];
		}
		test_counter++;
	}
	cout<<"Train Positive"<<endl;
	for (int i=0;i<INPUTS;i++) std::cout<<train_in1[0][i]<<" ";
	cout<<endl;
	for (int i=0;i<OUTPUTS;i++) std::cout<<train_out1[0][i]<<" ";
	cout<<endl;
	cout<<"Train Negative"<<endl;
	for (int i=0;i<INPUTS;i++) std::cout<<train_in0[0][i]<<" ";
	cout<<endl;
	for (int i=0;i<OUTPUTS;i++) std::cout<<train_out0[0][i]<<" ";
	cout<<endl;
	cout<<"Test"<<endl;
	for (int i=0;i<INPUTS;i++) std::cout<<test_in[0][i]<<" ";
	cout<<endl;
	for (int i=0;i<OUTPUTS;i++) std::cout<<test_out[0][i]<<" ";
	cout<<endl;	
}
StateNN::StateNN() {

	//set transfer function
	if (TRANSFER == 1) setDefaultTransferFunction(ANN::tanhFunction());
	else setDefaultTransferFunction(ANN::logisticFunction());

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
	trainer.setLearningRate(LEARNINGRATE);
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

TrainingPattern* getTrainingSampleBlanced(bool positive){
	TrainingPattern* p = new TrainingPattern;
	std::random_device rd; // obtain a random number from hardware
 	std::mt19937 eng(rd()); // seed the generator
	int max = 0;
	if (positive) max = SAMPLES1; 
	else max = SAMPLES0;
	std::uniform_real_distribution<> distr(0, (max-1)); // define the range
	int run_id =(int)distr(eng);
	for (int n=0;n<INPUTS;n++){
		if (positive) p->inputs[n] = train_in1[run_id][n];
		else p->inputs[n] = train_in0[run_id][n];
	}
	for (int n=0;n<OUTPUTS;n++){
		if (positive) p->outputs[n] = train_out1[run_id][n];
		else p->outputs[n] = train_out0[run_id][n];
	}
	return p;
}
