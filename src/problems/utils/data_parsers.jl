function libsvm_parser(path::String, n::Int64, d::Int64)::Data

    features = zeros(n, d)
    values = zeros(n)

    data_str = readlines(path)
    feature_value_str = split(data_str[2], " ")

    for i = 1:n
        feature_value_str = split(data_str[i], " ")
        value = parse(Float64, feature_value_str[1])
        values[i] = value

        for j = 3:(length(feature_value_str)-1)
            idx_feature_pair = split(feature_value_str[j], ':')

            idx = parse(Int64, idx_feature_pair[1])
            feature = parse(Float64, idx_feature_pair[2])
            features[i, idx] = feature
        end
    end
    Data(features, values)
end
