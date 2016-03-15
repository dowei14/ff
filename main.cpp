#include <utils/ann-framework/ann.h>
#include <utils/ann-framework/backpropagation.h>
#include <utils/ann-framework/neuron.h>
#include <iostream>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

using namespace std;
/*
class CSMTL : public ANN
{
	public:
		CSMTL();
		enum { nn_inputs = 15, nn_hidden = 20, nn_outputs=2 };
		
};

CSMTL::CSMTL() {

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
*/
class StateNN : public ANN
{
	public:
		StateNN();
		enum { nn_inputs = 11, nn_hidden = 20, nn_outputs=7 };
		
};

StateNN::StateNN() {

	//set transfer function
	setDefaultTransferFunction(ANN::tanhFunction());
//	setDefaultTransferFunction(ANN::logisticFunction());

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

	double train_in[60000][18];
	double train_out[60000][7];

	double test_in[60000][18];
	double test_out[60000][7];

void loadData(int inputs){
	ifstream data_inputs("../data/inRNN_11.txt"); //opening an input stream for file test.txt
	ifstream data_outputs("../data/outRNN_trigger.txt"); //opening an input stream for file test.txt
	std::string line_out;

	long trainer_counter = 0;
	for(std::string line_in; std::getline(data_inputs, line_in); )   //read stream line by line
	{		
		// add inputs
		std::istringstream in(line_in);      //make a stream for the line itself
		//cout<<"Inputs: ";
		for (int i=0;i<inputs;i++){
			in >> train_in[trainer_counter][i];                  //and read the first whitespace-separated token			
			//cout<<train_in[trainer_counter][i]<<" ";
		}

		// add outputs
		std::getline(data_outputs, line_out);
		std::istringstream out(line_out);
		//cout<<" Outputs: ";
		for (int o=0;o<7;o++){
			out >> train_out[trainer_counter][o];                  //and read the first whitespace-separated token		
			//cout<<train_out[trainer_counter][o]<<" ";
		}
		
		//cout<<endl;
		trainer_counter++;
	}

	ifstream data_inputs2("../data/inRNN_11.txt"); //opening an input stream for file test.txt
	ifstream data_outputs2("../data/outRNN_trigger.txt"); //opening an input stream for file test.txt
	long test_counter = 0;
	for(std::string line_in; std::getline(data_inputs2, line_in); )   //read stream line by line
	{
		
		// add inputs
		std::istringstream in(line_in);      //make a stream for the line itself
		//cout<<"Inputs: ";
		for (int i=0;i<inputs;i++){
			in >> test_in[test_counter][i];                  //and read the first whitespace-separated token
			
		}
		
		std::getline(data_outputs2, line_out);
		std::istringstream out(line_out);

		for (int o=0; o<7;o++){
			out >> test_out[test_counter][o];
		}
		test_counter++;
	}
	/*
	cout<<"Train"<<endl;
	for (int i=0;i<18;i++) std::cout<<train_in[0][i]<<" ";
	cout<<endl;
	for (int i=0;i<7;i++) std::cout<<train_out[0][i]<<" ";
	cout<<endl;
	cout<<"Test"<<endl;
	for (int i=0;i<18;i++) std::cout<<test_in[0][i]<<" ";
	cout<<endl;
	for (int i=0;i<7;i++) std::cout<<test_out[0][i]<<" ";
	cout<<endl;
	*/
}

int main(int argc, char **argv) {
	int input_size = 11;
	loadData(input_size);

	ofstream saveFile1;
	saveFile1.open("result.txt",ios::out);
	saveFile1.precision(5);
	saveFile1<<fixed;



for (int q=0;q<20;q++){
	cout<<"run: "<<q<<endl;
	StateNN ann;
	//cout<<ann.dumpWeights()<<endl;
	// create a topolocial sorting of our network. This is required for the
	// backpropagation algorithm as "Full Batch mode"
	ann.updateTopologicalSort();
	for (int a=0;a<100;a++){
		// create backpropagation object
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
		// create mini_batch	
		int mini_batch = 100;
		for (int i=0;i<mini_batch;i++){
			TrainingPattern* p = new TrainingPattern;
			std::random_device rd; // obtain a random number from hardware
	   	 	std::mt19937 eng(rd()); // seed the generator
			std::uniform_real_distribution<> distr(0, 32416); // define the range
			int run_id =(int)distr(eng);
			//cout<<run_id<<" ";
			for (int n=0;n<input_size;n++){
				p->inputs[n] = train_in[run_id][n];
			}
			for (int n=0;n<7;n++){
				p->outputs[n] = train_out[run_id][n];
			}
			trainer.addTrainingPattern(p);
		}
		//cout<<endl;

		// train
		trainer.learn(10);

		// test
		long errors = 0;
		for (long n=0;n<32416;n++){		
			for (int i=0;i<input_size;i++){
				ann.setInput(i,test_in[n][i]);
			}
			ann.feedForwardStep();
			double max = -9999;
			int maxID = 0;
			for (int i=0;i<7;i++){
				double val = ann.getOutput(i);
				if (val > max) {
					max = val;
					maxID = i;
				}
			}
			bool error = false;
			for (int i=0;i<7;i++){
				double val;
				if (i == maxID) val = 1.0;
				else val = 0.0;
				if (val != test_out[n][i]) error = true;
				//saveFile1 << val<<" ";
			}
			//saveFile1<<"\n";
			if (error) errors++;
		}
		cout<<(double)errors/32416<<endl;
		saveFile1 << (a+1)*10<< " " <<(double)errors/32416<<"\n";
		//cout<<ann.dumpWeights()<<endl;
	}
}
//	
//}
	

/*
  // create a topolocial sorting of our network. This is required for the
  // backpropagation algorithm as "Full Batch mode"
  ann.updateTopologicalSort();

  // training data
  const double data[4][3] = {{0, 0, 0},
                             {0, 1, 1},
                             {1, 0, 1},
                             {1, 1, 0}}; // {input 1, input 2, output}

  // create backpropagation object
  Backpropagation trainer;
  trainer.setNeuralNetwork(&ann);
  trainer.defineInputNeuron(0, ann.getNeuron(0));
  trainer.defineInputNeuron(1, ann.getNeuron(1));
  trainer.defineOutputNeuron(0, ann.getNeuron(4));
  trainer.includeAllSynapses();
  trainer.includeAllNeuronBiases();
  trainer.setLearningRate(0.1);
  for (int i=0; i<4; i++)
  {
    TrainingPattern* p = new TrainingPattern;
    p->inputs[0]  = data[i][0]; // text file here from your own created inputs
    p->inputs[1]  = data[i][1];
    p->outputs[0] = data[i][2];
    trainer.addTrainingPattern(p);
  }

  trainer.learn(50000);

  ann.setInput(0,0);
  ann.setInput(1,0);
  ann.feedForwardStep();
  std::cout << "0 0 => " << ann.getOutput(4) << std::endl;

  ann.setInput(0,1);
  ann.setInput(1,0);
  ann.feedForwardStep();
  std::cout << "1 0 => " << ann.getOutput(4) << std::endl;

  ann.setInput(0,0);
  ann.setInput(1,1);
  ann.feedForwardStep();
  std::cout << "0 1 => " << ann.getOutput(4) << std::endl;

  ann.setInput(0,1);
  ann.setInput(1,1);
  ann.feedForwardStep();
  std::cout << "1 1 => " << ann.getOutput(4) << std::endl;

  std::cout << ann.dumpWeights();
  */
}

