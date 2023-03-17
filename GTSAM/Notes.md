**GTSAM notes:**
- Values để lưu biến phục vụ quá trình optimize
- Các phương pháp tối ưu: LevenbergMarquardt, Powel's Dogleg, GaussNewton
- Các params được setup trước khi tối ưu (lưu ý các mặc định Params của GTSAM)
- Keys and Symbols là cách thức định nghĩa biến của GTSAM

**SFM examples:** 
- Data association is known
- Graph is constructed -> ok fine
- Tạo (i) poses và (j) points/landmarks có chủ đích để test (done)
- Tạo các measures với nhiễu -> project về camera poses
- Tạo Loop để định hình FactorGraph giữa:
  - (i) pose-nodes X(i) 
  - (j) lm-nodes L(j)
  - Định nghĩa error_function giữa X(i) và L(j):
    - Các measurement & measurement_noise
    - Các tham số phụ trợ (K - calibrations)
- Tạo data structure để lưu lời giải của giải thuật:
  - Lưu tọa độ points (với nhiễu 0.1) với Key là X(i)
  - Lưu poses (với nhiễu 0.1) với Key lầ L(j)
