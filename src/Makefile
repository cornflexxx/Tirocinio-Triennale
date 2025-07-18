NVCC = nvcc -ccbin mpic++

NVCC_FLAGS = -std=c++11

INCLUDES = -I../include


SRC = GSZ_entry.cu GSZ_timer.cu comprs_test.cu GSZ.cu  main_hom.cu 
TARGET = main_hom

SRC_2 = GSZ_entry.cu GSZ_timer.cu comprs_test.cu GSZ.cu  main_allred.cu mpi_awareallreduce.cu
TARGET_2 = main_allred

SRC_3 = GSZ_entry.cu GSZ_timer.cu comprs_test.cu GSZ.cu  main_naw_allred.cu nca_allreduce.cu
TARGET_3 = nca_allred

compression: $(TARGET)

aw_allreduce: $(TARGET_2)

naw_allreduce: $(TARGET_3)

$(TARGET_2): $(SRC_2)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRC_2) -o $(TARGET_2) -lcudart

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRC) -o $(TARGET) -lcudart

$(TARGET_3): $(SRC_3)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRC_3) -o $(TARGET_3) -lcudart

clean:
	rm -f $(TARGET) $(TARGET_2) $(TARGET_3)
