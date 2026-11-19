#include <iostream>
#include <numeric>
#include <thread>
#include <atomic>
#include <chrono>
#include <inttypes.h>

#include <gegelati.h>

#include "mnist_reader/mnist_reader.hpp"

int main() {
	std::cout << "Start MNIST application." << std::endl;

	// Create the instruction set for programs
	Instructions::Set set;
	auto minus = [](double a, double b)->double {return a - b; };
	auto add = [](double a, double b)->double {return a + b; };
	auto mult = [](double a, double b)->double {return a * b; };
	auto div = [](double a, double b)->double {return a / b; };
	auto max = [](double a, double b)->double {return std::max(a, b); };
	auto ln = [](double a)->double {return std::log(a); };
	auto exp = [](double a)->double {return std::exp(a); };
	auto sobelMagn = [](const double a[3][3])->double {
		double result = 0.0;
		double gx =
			-a[0][0] + a[0][2]
			- 2.0 * a[1][0] + 2.0 * a[1][2]
			- a[2][0] + a[2][2];
		double gy = -a[0][0] - 2.0 * a[0][1] - a[0][2]
			+ a[2][0] + 2.0 * a[2][1] + a[2][2];
		result = sqrt(gx * gx + gy * gy);
		return result;
	};

	auto sobelDir = [](const double a[3][3])->double {
		double result = 0.0;
		double gx =
			-a[0][0] + a[0][2]
			- 2.0 * a[1][0] + 2.0 * a[1][2]
			- a[2][0] + a[2][2];
		double gy = -a[0][0] - 2.0 * a[0][1] - a[0][2]
			+ a[2][0] + 2.0 * a[2][1] + a[2][2];
		result = std::atan(gy / gx);
		return result;
	};
	
	set.add(*(new Instructions::LambdaInstruction<double, double, double>(minus)));
	set.add(*(new Instructions::LambdaInstruction<double, double, double>(add)));
	set.add(*(new Instructions::LambdaInstruction<double, double, double>(mult)));
	set.add(*(new Instructions::LambdaInstruction<double, double, double>(div)));
	set.add(*(new Instructions::LambdaInstruction<double, double, double>(max)));
	set.add(*(new Instructions::LambdaInstruction<double, double>(exp)));
	set.add(*(new Instructions::LambdaInstruction<double, double>(ln)));
	set.add(*(new Instructions::LambdaInstruction<double, const double[3][3]>(sobelMagn)));
	set.add(*(new Instructions::LambdaInstruction<double, const double[3][3]>(sobelDir)));

	/// Build MNIST
    mnist::MNIST_dataset<std::vector, std::vector<double>, uint8_t> mnist(mnist::read_dataset<std::vector, std::vector, double, uint8_t>(MNIST_DATA_LOCATION));
    Dimensions::Requirement inputReq = Dimensions::Requirement(Data::DataType::type<double>({28, 28}));
    Dimensions::Requirement outputReq = Dimensions::Requirement(Data::DataType::type<size_t>());
    Evaluation::DataSet dataSet({inputReq}, outputReq);

	/// Add training images
    std::vector<std::vector<double>> trainingImages = mnist.training_images;
    std::vector<uint8_t> trainingLabels = mnist.training_labels;
    for(size_t idx = 0; idx < trainingImages.size(); idx++) {
        std::vector<Data::DataValue> inputs;
        inputs.push_back(Data::DataValue::array2d<std::vector<double>>(trainingImages.at(idx), 28, 28));
        dataSet.addData(std::move(inputs), Data::DataValue::scalar<uint64_t>(static_cast<uint64_t>(trainingLabels.at(idx))));
    }

	/// Add validation images
    std::vector<std::vector<double>> testImages = mnist.test_images;
    std::vector<uint8_t> testLabels = mnist.test_labels;
    for(size_t idx = 0; idx < testImages.size(); idx++) {
        std::vector<Data::DataValue> inputs;
        inputs.push_back(Data::DataValue::array2d<std::vector<double>>(testImages.at(idx), 28, 28));
        dataSet.addData(std::move(inputs), Data::DataValue::scalar<uint64_t>(static_cast<uint64_t>(testLabels.at(idx))), false);
    }

	auto problem = std::make_shared<Evaluation::PredictionProblem>(dataSet, 0, "MNIST");


	// for building MNIST with LGP
	/*
	auto representation = std::make_shared<Representations::LGP>(problem->getInputDimensions(), Data::DataType::type<double>({15}), set, Data::DataType::type<double>({10}), 1, 20, "MyLGP");
	auto tanh = std::make_unique<Dimensions::ActivationFunctions::ArgMax<double>>(representation->getDimensionFlow().getOutputDimension());
	representation->addOutputFunction(std::move(tanh));

	Evolution::BasicEvolutionConfig config;
	*/


	// For building MNIST with TPG
	/**/
	auto lgpRep = std::make_shared<Representations::LGP>(problem->getInputDimensions(), Data::DataType::type<double>({8}), set, Data::DataType::type<double>({1}), 1, 20, "MyLGP");
	auto tpgRep = std::make_shared<Representations::TPG>(problem->getInputDimensions(), 10, 2, 10, "MyTPG");

	Evolution::OriginTPGEvolutionConfig config;
	config.memberRepresentation = lgpRep;
	config.teamDestinationProbability = 0.5;


	config.problem = problem;
	config.representation = tpgRep;
	config.nbTrainingIterations = 100;
	config.nbValidationIterations = 100;
	config.sizePopulation = 100;
	config.nbOffspring = 200;
	config.nbThreads = 2;
	
	// In classification, you want the F1 metric
	// config.assigner = std::make_shared<FitnessAssignment::ClassificationF1Assigner>();

	std::shared_ptr<Evolution::Module> module = config.buildModule();

	std::cout<<module->summary(1)<<std::endl;
	Evolution::EvolutionData data = module->evolve(100, 2);
	std::cout<<data.summary()<<std::endl;

	return 0;
}
