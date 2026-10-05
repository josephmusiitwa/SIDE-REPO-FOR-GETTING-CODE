#include "meanshift/Dataset.hpp"
#include "meanshift/MeanShift.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <filesystem>
#include <fstream>
namespace fs = std::filesystem;

int main() {
    std::string inputFolder = "data/input";
    std::string outputFolder = "data/output";
    
    std::cout << "--- Auto-Scanning Batch Processor ---" << std::endl;
    std::cout << "Scanning folder: " << inputFolder << " for .csv files..." << std::endl;

    // Check if the input directory actually exists
    if (!fs::exists(inputFolder) || !fs::is_directory(inputFolder)) {
        std::cerr << "Error: Directory '" << inputFolder << "' does not exist." << std::endl;
        return 1;
    }

    int filesProcessed = 0;

    // Magically loop through every single file inside the input folder
    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        
        // We only care about normal files that end in ".csv"
        if (entry.is_regular_file() && entry.path().extension() == ".csv") {
            
            std::string currentInputFile = entry.path().string();
            std::string baseFileName = entry.path().stem().string(); // grabs just the name (e.g., "my_data")
            
            // Automatically generate a matching output file name!
            std::string currentOutputFile = outputFolder + "/results_" + baseFileName + ".csv";
            
            std::cout << "\n========================================" << std::endl;
            std::cout << "Processing File: " << baseFileName << ".csv" << std::endl;
            
            meanshift::Dataset dataset;
            if (!dataset.loadFromCSV(currentInputFile, true)) {
                std::cerr << "-> Skipping... (Failed to read CSV)" << std::endl;
                continue; 
            }
            
            std::cout << "-> Running MeanShift clustering..." << std::endl;
            meanshift::MeanShift ms(2.0, 1e-3, 300);
            ms.fit(dataset.getData());
            
            std::cout << "-> Number of clusters found: " << ms.getClusters().size() << std::endl;
            std::cout << "-> Saving to: " << currentOutputFile << std::endl;
            
            std::ofstream outFile(currentOutputFile);
            if (outFile.is_open()) {
                outFile << "Feature1,Feature2,Cluster_ID\n";
                const auto& realData = dataset.getData();
                const auto& labels = ms.getLabels();
                
                for (std::size_t k = 0; k < realData.size(); ++k) {
                    for (std::size_t d = 0; d < realData[k].size(); ++d) {
                        outFile << realData[k][d] << ",";
                    }
                    outFile << labels[k] << "\n";
                }
                outFile.close();
                std::cout << "-> Success!" << std::endl;
                filesProcessed++;
            } else {
                std::cerr << "-> Warning: Could not open output file for writing!" << std::endl;
            }
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Finished! " << filesProcessed << " files were successfully processed." << std::endl;

    return 0;
}
