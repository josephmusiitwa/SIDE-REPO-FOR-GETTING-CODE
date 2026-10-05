#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>

using namespace std;

namespace meanshift {

using Point = vector<double>;
using Data = vector<Point>;

class Dataset {
public:
    Dataset();

    // Dataset loading from CSV
    bool loadFromCSV(const string& filename, bool hasHeader = true);

    // Get the raw data
    const Data& getData() const;

    // Feature scaling: Standard Scaler (z-score normalization)
    void standardize();

    // Feature scaling: Min-Max Scaler
    void normalize();

    // Data preprocessing (remove NaNs, etc., simple version removes rows with invalid values if any)
    // Note: Assuming valid numeric CSV for simplicity.
private:
    Data data_;
};

} // namespace meanshift

#endif // DATASET_HPP
