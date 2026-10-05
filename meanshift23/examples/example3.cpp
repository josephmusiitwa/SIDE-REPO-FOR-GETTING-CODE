#include "meanshift/Dataset.hpp"
#include "meanshift/MeanShift.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

using namespace std;

int main(int argc, char* argv[]) {
    // 1. Specify the input file
    string csvFilePath = "data/input/my_data.csv";
    if (argc > 1) {
        csvFilePath = argv[1];
    }
    
    meanshift::Dataset dataset;
    cout << "Loading data from external CSV file: " << csvFilePath << endl;
    
    // 2. Load the external CSV data
    if (!dataset.loadFromCSV(csvFilePath, true)) {
        cerr << "Error: Failed to load external data." << endl;
        return 1;
    }
    
    cout << "Successfully loaded external data." << endl;
    cout << "Running MeanShift clustering..." << endl;
    
    // 3. Set up MeanShift algorithm and fit the data
    meanshift::MeanShift ms(2.0, 1e-3, 300);
    ms.fit(dataset.getData());
    
    // 4. Output the results ONLY to the terminal
    auto clusters = ms.getClusters();
    cout << "\nNumber of clusters found: " << clusters.size() << endl;
    
    int i = 0;
    for (const auto& c : clusters) {
        cout << "Cluster " << i++ << " centroid: (";
        for (size_t d = 0; d < c.centroid.size(); ++d) {
            cout << c.centroid[d] << (d == c.centroid.size() - 1 ? "" : ", ");
        }
        cout << ") with " << c.pointIndices.size() << " points." << endl;
    }
    
    cout << "Clustering evaluation score: " << ms.evaluate(dataset.getData()) << endl;
    
    // We intentionally stop here without writing anything to data/output/
    return 0;
}
