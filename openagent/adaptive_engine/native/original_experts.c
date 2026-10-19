#ifndef _WIN32
#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#endif
#include "original_experts.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdatomic.h>
#include <limits.h>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif
typedef struct {FILE *stream;uint64_t size;int64_t mtime_ns;} original_file;
struct oa_original_experts {
    atomic_flag guard;atomic_uint_fast64_t refs;uint64_t hidden,width,bytes;
    uint32_t file_count,expert_count;float limit;
    original_file *files;oa_original_expert *experts;
    unsigned char *raw,*scales;float *gate,*up,*row;
};
struct oa_original_projections {
    atomic_flag guard;atomic_uint_fast64_t refs;
    oa_original_experts *storage;oa_original_matrix *matrices;
    uint32_t count,row_only;uint64_t bytes;
};
static int fail(char *error,size_t bytes,const char *message) {
    if(error&&bytes)snprintf(error,bytes,"%s",message);return -1;
}
static unsigned dtype_width(uint32_t dtype) {return dtype==1?4:(dtype==2||dtype==3?2:(dtype==4||dtype==5?1:0));}
static int file_stamp(FILE *stream,uint64_t *size,int64_t *mtime) {
#ifdef _WIN32
    BY_HANDLE_FILE_INFORMATION info;ULARGE_INTEGER time,length;
    HANDLE handle=(HANDLE)_get_osfhandle(_fileno(stream));
    if(handle==INVALID_HANDLE_VALUE||!GetFileInformationByHandle(handle,&info))return 0;
    length.HighPart=info.nFileSizeHigh;length.LowPart=info.nFileSizeLow;
    time.HighPart=info.ftLastWriteTime.dwHighDateTime;time.LowPart=info.ftLastWriteTime.dwLowDateTime;
    if(time.QuadPart<UINT64_C(116444736000000000)||time.QuadPart-UINT64_C(116444736000000000)>INT64_MAX/100)return 0;
    *size=length.QuadPart;*mtime=(int64_t)(time.QuadPart-UINT64_C(116444736000000000))*100;return 1;
#else
    struct stat status;if(fstat(fileno(stream),&status)||status.st_size<0)return 0;
    *size=(uint64_t)status.st_size;
#ifdef __APPLE__
    *mtime=(int64_t)status.st_mtimespec.tv_sec*INT64_C(1000000000)+status.st_mtimespec.tv_nsec;
#else
    *mtime=(int64_t)status.st_mtim.tv_sec*INT64_C(1000000000)+status.st_mtim.tv_nsec;
#endif
    return 1;
#endif
}
static FILE *open_file(const char *path) {
#ifdef _WIN32
    int count;wchar_t *wide;FILE *stream;
    if(!path)return NULL;
    count=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,NULL,0);
    if(count<=0||count>32768)return NULL;
    wide=(wchar_t *)malloc((size_t)count*sizeof(wchar_t));if(!wide)return NULL;
    if(!MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,wide,count)){free(wide);return NULL;}
    stream=_wfopen(wide,L"rb");free(wide);return stream;
#else
    return path?fopen(path,"rb"):NULL;
#endif
}
static int unchanged(original_file *file) {
    uint64_t size;int64_t mtime;
    return file_stamp(file->stream,&size,&mtime)&&size==file->size&&mtime==file->mtime_ns;
}
static int read_at(original_file *file,uint64_t offset,void *output,size_t count) {
    if(offset>INT64_MAX||offset>file->size||count>file->size-offset||!unchanged(file))return 0;
#ifdef _WIN32
    if(_fseeki64(file->stream,(int64_t)offset,SEEK_SET))return 0;
#else
    if(fseeko(file->stream,(off_t)offset,SEEK_SET))return 0;
#endif
    return fread(output,1,count,file->stream)==count&&unchanged(file);
}
static float decode(const unsigned char *data,uint32_t dtype) {
    uint32_t bits;float value;
    if(dtype==1)bits=(uint32_t)data[0]|(uint32_t)data[1]<<8|(uint32_t)data[2]<<16|(uint32_t)data[3]<<24;
    else if(dtype==3)bits=((uint32_t)data[0]|(uint32_t)data[1]<<8)<<16;
    else if(dtype==2) {
        uint32_t half=(uint32_t)data[0]|(uint32_t)data[1]<<8,exponent=(half>>10)&31,fraction=half&1023;
        if(exponent==31)return NAN;
        value=exponent?ldexpf(1.0f+(float)fraction/1024,(int)exponent-15):ldexpf((float)fraction,-24);
        return half&32768?-value:value;
    } else if(dtype==4) {
        unsigned byte=data[0],exponent=(byte>>3)&15,fraction=byte&7;
        if(exponent==15&&fraction==7)return NAN;
        value=exponent?ldexpf(1+(float)fraction/8,(int)exponent-7):ldexpf((float)fraction,-9);
        return byte&128?-value:value;
    } else if(dtype==5) {
        if(data[0]==255)return NAN;
        if(!data[0])return ldexpf(1.0f,-127);
        bits=(uint32_t)data[0]<<23;
    } else return NAN;
    memcpy(&value,&bits,4);return value;
}
static int matrix_valid(oa_original_experts *bank,const oa_original_matrix *matrix,uint64_t rows,uint64_t columns) {
    uint64_t count,scale_rows,scale_columns,matrices;unsigned width=dtype_width(matrix->dtype);
    if(matrix->file>=bank->file_count||matrix->dtype<1||matrix->dtype>4||!width||
       matrix->rows!=rows||matrix->columns!=columns||!matrix->parent_rows||
       matrix->parent_rows>UINT32_MAX||matrix->parent_rows>UINT64_MAX/columns||
       rows>UINT64_MAX/columns)return 0;
    count=rows*columns;
    if(matrix->element_offset>matrix->elements||count>matrix->elements-matrix->element_offset||
       matrix->element_offset%columns||matrix->elements>UINT64_MAX/width||
       matrix->offset>bank->files[matrix->file].size||
       matrix->elements*width>bank->files[matrix->file].size-matrix->offset)return 0;
    if(matrix->dtype!=4)return matrix->scale_elements==0;
    if(matrix->scale_file>=bank->file_count||!dtype_width(matrix->scale_dtype)||
       matrix->scale_dtype==4||!matrix->block_rows||!matrix->block_columns||
       matrix->block_rows>4096||matrix->block_columns>4096||
       matrix->elements%(matrix->parent_rows*columns))return 0;
    scale_rows=(matrix->parent_rows+matrix->block_rows-1)/matrix->block_rows;
    scale_columns=(columns+matrix->block_columns-1)/matrix->block_columns;
    matrices=matrix->elements/(matrix->parent_rows*columns);
    if(matrices>UINT64_MAX/scale_rows||matrices*scale_rows>UINT64_MAX/scale_columns||
       matrix->scale_elements!=matrices*scale_rows*scale_columns||
       matrix->scale_elements>UINT64_MAX/dtype_width(matrix->scale_dtype)||
       matrix->scale_offset>bank->files[matrix->scale_file].size||
       matrix->scale_elements*dtype_width(matrix->scale_dtype)>bank->files[matrix->scale_file].size-matrix->scale_offset)return 0;
    return 1;
}
static int project(oa_original_experts *bank,const oa_original_matrix *matrix,const float *input,float *output,
    int (*cancel)(void *),void *cancel_data,int fp32_reduction) {
    uint64_t row,column;unsigned width=dtype_width(matrix->dtype);
    for(row=0;row<matrix->rows;row++) {
        if(cancel&&cancel(cancel_data))return 1;
        uint64_t first=matrix->element_offset+row*matrix->columns;double sum=0;float factor=1;
        if(!read_at(bank->files+matrix->file,matrix->offset+first*width,bank->raw,(size_t)matrix->columns*width))return 1;
        if(matrix->dtype==4) {
            uint64_t original_row=first/matrix->columns;
            uint64_t scale_columns=(matrix->columns+matrix->block_columns-1)/matrix->block_columns;
            uint64_t scale_rows=(matrix->parent_rows+matrix->block_rows-1)/matrix->block_rows;
            uint64_t index=(original_row/matrix->parent_rows)*scale_rows*scale_columns+
                (original_row%matrix->parent_rows/matrix->block_rows)*scale_columns;
            unsigned scale_width=dtype_width(matrix->scale_dtype);
            if(!read_at(bank->files+matrix->scale_file,matrix->scale_offset+index*scale_width,
                bank->scales,(size_t)scale_columns*scale_width))return 1;
        }
        for(column=0;column<matrix->columns;column++) {
            float value=decode(bank->raw+column*width,matrix->dtype);
            if(matrix->dtype==4 && column%matrix->block_columns==0) {
                unsigned scale_width=dtype_width(matrix->scale_dtype);
                factor=decode(bank->scales+(column/matrix->block_columns)*scale_width,matrix->scale_dtype);
                if(!isfinite(factor)||factor<=0)return 1;
            }
            value*=factor;if(!isfinite(value))return 1;
            if(fp32_reduction)sum=(double)((float)sum+value*input[column]);
            else sum+=(double)value*input[column];
        }
        if(!isfinite(sum)||!isfinite((float)sum))return 1;output[row]=(float)sum;
    }
    return 0;
}
static void destroy(oa_original_experts *bank) {
    uint32_t i;if(!bank)return;
    if(bank->files)for(i=0;i<bank->file_count;i++)if(bank->files[i].stream)fclose(bank->files[i].stream);
    free(bank->files);free(bank->experts);free(bank->raw);free(bank->scales);free(bank->gate);free(bank->up);free(bank->row);free(bank);
}
uint32_t oa_original_experts_version(void){return 1;}
uint32_t oa_original_projections_version(void){return 1;}
static int original_projections_create(const oa_original_file *files,uint32_t file_count,
    const oa_original_matrix *matrices,uint32_t count,uint64_t max_bytes,
    oa_original_projections **out,char *error,size_t error_bytes,uint32_t row_only) {
    oa_original_projections *owner=NULL;oa_original_experts *storage=NULL;
    uint64_t columns=0,rows=0,bytes;uint32_t i;uint16_t endian=1;
    if(out)*out=NULL;
    if(!files||!matrices||!out||!file_count||file_count>4096||!count||count>1048576||
       sizeof(float)!=4||*(unsigned char *)&endian!=1)
        return fail(error,error_bytes,"original dense projection inventory differs");
    for(i=0;i<count;i++) {
        if(!matrices[i].rows||!matrices[i].columns||matrices[i].rows>(row_only?UINT32_MAX:1048576)||matrices[i].columns>1048576)
            return fail(error,error_bytes,"original dense projection geometry exceeds admission");
        if(matrices[i].rows>rows)rows=matrices[i].rows;
        if(matrices[i].columns>columns)columns=matrices[i].columns;
    }
    if(row_only)rows=1;
    bytes=sizeof(*owner)+sizeof(*storage)+(uint64_t)file_count*sizeof(original_file)+
        (uint64_t)count*sizeof(*matrices)+4*(3*columns+rows);
    if(bytes>max_bytes||bytes>SIZE_MAX)
        return fail(error,error_bytes,"original dense projection aggregate admission exhausted");
    owner=(oa_original_projections *)calloc(1,sizeof(*owner));
    storage=(oa_original_experts *)calloc(1,sizeof(*storage));
    if(!owner||!storage)goto invalid;
    owner->storage=storage;owner->count=count;owner->bytes=bytes;owner->row_only=row_only;
    atomic_flag_clear(&owner->guard);atomic_init(&owner->refs,1);
    storage->file_count=file_count;
    storage->files=(original_file *)calloc(file_count,sizeof(original_file));
    owner->matrices=(oa_original_matrix *)malloc((size_t)count*sizeof(*matrices));
    storage->raw=(unsigned char *)malloc((size_t)columns*4);
    storage->scales=(unsigned char *)malloc((size_t)columns*4);
    storage->row=(float *)malloc((size_t)rows*4);
    storage->gate=(float *)malloc((size_t)columns*4);
    if(!storage->files||!owner->matrices||!storage->raw||!storage->scales||!storage->row||!storage->gate)goto invalid;
    for(i=0;i<file_count;i++) {
        storage->files[i].stream=open_file(files[i].path);
        storage->files[i].size=files[i].size;storage->files[i].mtime_ns=files[i].mtime_ns;
        if(!storage->files[i].stream||!unchanged(storage->files+i)||
           setvbuf(storage->files[i].stream,NULL,_IONBF,0))goto invalid;
    }
    memcpy(owner->matrices,matrices,(size_t)count*sizeof(*matrices));
    for(i=0;i<count;i++)
        if(!matrix_valid(storage,owner->matrices+i,matrices[i].rows,matrices[i].columns))goto invalid;
    *out=owner;return 0;
invalid:
    if(storage)destroy(storage);
    if(owner){free(owner->matrices);free(owner);}
    return fail(error,error_bytes,"original dense projection files or storage differ");
}
int oa_original_projections_create(const oa_original_file *files,uint32_t file_count,
    const oa_original_matrix *matrices,uint32_t count,uint64_t maximum,
    oa_original_projections **out,char *error,size_t error_bytes) {
    return original_projections_create(files,file_count,matrices,count,maximum,out,error,error_bytes,0);
}
int oa_original_projections_create_rows(const oa_original_file *files,uint32_t file_count,
    const oa_original_matrix *matrices,uint32_t count,uint64_t maximum,
    oa_original_projections **out,char *error,size_t error_bytes) {
    return original_projections_create(files,file_count,matrices,count,maximum,out,error,error_bytes,1);
}
int oa_original_projections_retain(oa_original_projections *owner) {
    uint_fast64_t refs;if(!owner)return -1;refs=atomic_load(&owner->refs);
    do{if(!refs||refs==UINT64_MAX)return -1;}
    while(!atomic_compare_exchange_weak(&owner->refs,&refs,refs+1));return 0;
}
int oa_original_projections_release(oa_original_projections *owner) {
    uint_fast64_t refs;if(!owner)return -1;refs=atomic_load(&owner->refs);
    do{if(!refs)return -1;}
    while(!atomic_compare_exchange_weak(&owner->refs,&refs,refs-1));
    if(refs==1){destroy(owner->storage);free(owner->matrices);free(owner);}return 0;
}
int oa_original_projection_shape(oa_original_projections *owner,uint32_t index,uint64_t *rows,uint64_t *columns) {
    if(!owner||!rows||!columns||index>=owner->count||oa_original_projections_retain(owner))return -1;
    *rows=owner->matrices[index].rows;*columns=owner->matrices[index].columns;
    oa_original_projections_release(owner);return 0;
}
int oa_original_projection_evaluate_policy(oa_original_projections *owner,uint32_t index,
    const float *input,uint64_t input_count,float *output,uint64_t output_count,
    int (*cancel)(void *),void *cancel_data,uint32_t fp32_reduction) {
    uint64_t i;int status=1;oa_original_matrix *matrix;
    if(!owner||!input||!output||fp32_reduction>1||oa_original_projections_retain(owner))return 1;
    if(owner->row_only){oa_original_projections_release(owner);return 1;}
    if(atomic_flag_test_and_set(&owner->guard)){oa_original_projections_release(owner);return 1;}
    if(index>=owner->count)goto done;
    matrix=owner->matrices+index;
    if(input_count!=matrix->columns||output_count!=matrix->rows)goto done;
    for(i=0;i<input_count;i++)if(!isfinite(input[i]))goto done;
    if(project(owner->storage,matrix,input,owner->storage->row,cancel,cancel_data,(int)fp32_reduction))goto done;
    if(cancel&&cancel(cancel_data))goto done;
    memcpy(output,owner->storage->row,(size_t)matrix->rows*sizeof(float));status=0;
done:atomic_flag_clear(&owner->guard);oa_original_projections_release(owner);return status;
}
int oa_original_projection_evaluate(oa_original_projections *owner,uint32_t index,
    const float *input,uint64_t input_count,float *output,uint64_t output_count,
    int (*cancel)(void *),void *cancel_data) {
    return oa_original_projection_evaluate_policy(owner,index,input,input_count,output,output_count,cancel,cancel_data,1);
}
int oa_original_projection_read_row(oa_original_projections *owner,uint32_t index,
    uint64_t row,float *output,uint64_t count) {
    oa_original_matrix *m;oa_original_experts *s;uint64_t first,column;unsigned width;int status=1;
    if(!owner||!output||oa_original_projections_retain(owner))return 1;
    if(atomic_flag_test_and_set(&owner->guard)){oa_original_projections_release(owner);return 1;}
    if(index>=owner->count)goto done;m=owner->matrices+index;s=owner->storage;
    if(row>=m->rows||count!=m->columns)goto done;
    first=m->element_offset+row*m->columns;width=dtype_width(m->dtype);
    if(!read_at(s->files+m->file,m->offset+first*width,s->raw,(size_t)m->columns*width))goto done;
    if(m->dtype==4) {
        uint64_t original=first/m->columns,cols=(m->columns+m->block_columns-1)/m->block_columns;
        uint64_t rows=(m->parent_rows+m->block_rows-1)/m->block_rows;
        uint64_t scale=(original/m->parent_rows)*rows*cols+(original%m->parent_rows/m->block_rows)*cols;
        unsigned sw=dtype_width(m->scale_dtype);
        if(!read_at(s->files+m->scale_file,m->scale_offset+scale*sw,s->scales,(size_t)cols*sw))goto done;
    }
    for(column=0;column<count;column++) {
        float factor=m->dtype==4?decode(s->scales+(column/m->block_columns)*dtype_width(m->scale_dtype),m->scale_dtype):1;
        float value=decode(s->raw+column*width,m->dtype)*factor;
        /* Match PublisherTensorReader and project(): FP8 block scales are
         * strictly positive, not merely finite.  A zero/negative scale in a
         * row-table read must not silently publish a different interpretation
         * than streamed projection evaluation. */
        if(!isfinite(value)||!isfinite(factor)||factor<=0.0f)goto done;s->gate[column]=value;
    }
    memcpy(output,s->gate,(size_t)count*sizeof(float));status=0;
done:atomic_flag_clear(&owner->guard);oa_original_projections_release(owner);return status;
}
struct oa_original_rows {
    atomic_uint_fast64_t refs;
    oa_original_projections *owner;uint32_t count;
    uint32_t *indices;uint64_t *starts,columns;
};
int oa_original_rows_create(oa_original_projections *owner,const uint32_t *indices,uint32_t count,
    uint64_t maximum,oa_original_rows **out,char *error,size_t error_bytes) {
    oa_original_rows *table;uint32_t i;uint64_t rows,columns,bytes;
    if(out)*out=NULL;
    if(!owner||!indices||!count||count>4096||!out)return fail(error,error_bytes,"original row table inventory differs");
    bytes=sizeof(*table)+(uint64_t)count*4+((uint64_t)count+1)*8;
    if(bytes>maximum||oa_original_projections_retain(owner))return fail(error,error_bytes,"original row table admission exhausted");
    table=(oa_original_rows *)calloc(1,sizeof(*table));
    if(!table){oa_original_projections_release(owner);return fail(error,error_bytes,"original row table allocation failed");}
    table->owner=owner;table->count=count;
    atomic_init(&table->refs,1);
    table->indices=(uint32_t *)malloc((size_t)count*4);table->starts=(uint64_t *)calloc((size_t)count+1,8);
    if(!table->indices||!table->starts)goto invalid;
    for(i=0;i<count;i++) {
        if(oa_original_projection_shape(owner,indices[i],&rows,&columns)||
            (i&&columns!=table->columns)||rows>UINT64_MAX-table->starts[i])goto invalid;
        table->indices[i]=indices[i];table->starts[i+1]=table->starts[i]+rows;table->columns=columns;
    }
    *out=table;return 0;
invalid:oa_original_rows_release(table);return fail(error,error_bytes,"original row table geometry differs");
}
int oa_original_rows_release(oa_original_rows *table) {
    uint_fast64_t refs;if(!table)return -1;refs=atomic_load(&table->refs);
    do{if(!refs)return -1;}while(!atomic_compare_exchange_weak(&table->refs,&refs,refs-1));
    if(refs>1)return 0;oa_original_projections_release(table->owner);
    free(table->indices);free(table->starts);free(table);return 0;
}
int oa_original_rows_retain(oa_original_rows *table) {
    uint_fast64_t refs;if(!table)return -1;refs=atomic_load(&table->refs);
    do{if(!refs||refs==UINT64_MAX)return -1;}while(!atomic_compare_exchange_weak(&table->refs,&refs,refs+1));return 0;
}
int oa_original_rows_shape(oa_original_rows *table,uint64_t *rows,uint64_t *columns) {
    if(!table||!rows||!columns||oa_original_rows_retain(table))return -1;
    *rows=table->starts[table->count];*columns=table->columns;oa_original_rows_release(table);return 0;
}
int oa_original_rows_lookup(void *context,uint64_t row,uint64_t columns,float *output) {
    oa_original_rows *table=(oa_original_rows *)context;uint32_t left=0,right,middle;
    if(!table||columns!=table->columns||row>=table->starts[table->count])return 1;
    right=table->count;
    while(left+1<right){middle=left+(right-left)/2;if(table->starts[middle]<=row)left=middle;else right=middle;}
    return oa_original_projection_read_row(table->owner,table->indices[left],row-table->starts[left],output,columns);
}
int oa_original_experts_create(const oa_original_file *files,uint32_t file_count,
    const oa_original_expert *experts,uint32_t expert_count,uint64_t hidden,uint64_t width,
    float limit,uint64_t max_bytes,oa_original_experts **out,char *error,size_t error_bytes) {
    oa_original_experts *bank;uint64_t bytes,maximum;uint32_t i;uint16_t endian=1;
    if(out)*out=NULL;
    if(!files||!experts||!out||!file_count||file_count>4096||!expert_count||expert_count>1048576||
       !hidden||hidden>1048576||!width||width>1048576||!isfinite(limit)||limit<=0||
       sizeof(float)!=4||*(unsigned char *)&endian!=1)return fail(error,error_bytes,"native original expert descriptor differs");
    maximum=hidden>width?hidden:width;
    bytes=sizeof(*bank)+(uint64_t)file_count*sizeof(original_file)+(uint64_t)expert_count*sizeof(*experts)+
        4*(2*maximum+3*width+hidden);
    if(bytes>max_bytes||bytes>SIZE_MAX)return fail(error,error_bytes,"native original expert aggregate admission exhausted");
    bank=(oa_original_experts *)calloc(1,sizeof(*bank));if(!bank)return fail(error,error_bytes,"native original expert allocation failed");
    atomic_flag_clear(&bank->guard);atomic_init(&bank->refs,1);bank->hidden=hidden;bank->width=width;bank->limit=limit;bank->bytes=bytes;
    bank->file_count=file_count;bank->expert_count=expert_count;
    bank->files=(original_file *)calloc(file_count,sizeof(*bank->files));bank->experts=(oa_original_expert *)malloc((size_t)expert_count*sizeof(*experts));
    bank->raw=(unsigned char *)malloc((size_t)maximum*4);bank->gate=(float *)malloc((size_t)width*4);
    bank->scales=(unsigned char *)malloc((size_t)maximum*4);
    bank->up=(float *)malloc((size_t)width*4);bank->row=(float *)malloc((size_t)hidden*4);
    if(!bank->files||!bank->experts||!bank->raw||!bank->scales||!bank->gate||!bank->up||!bank->row)goto invalid;
    for(i=0;i<file_count;i++) {
        bank->files[i].stream=open_file(files[i].path);bank->files[i].size=files[i].size;bank->files[i].mtime_ns=files[i].mtime_ns;
        if(!bank->files[i].stream||!unchanged(bank->files+i))goto invalid;
        if(setvbuf(bank->files[i].stream,NULL,_IONBF,0))goto invalid;
    }
    memcpy(bank->experts,experts,(size_t)expert_count*sizeof(*experts));
    for(i=0;i<expert_count;i++) {
        oa_original_expert *entry=bank->experts+i;
        if(entry->layer>INT32_MAX||entry->expert>INT32_MAX||
           (i&&((uint64_t)bank->experts[i-1].layer<<32|bank->experts[i-1].expert)>=((uint64_t)entry->layer<<32|entry->expert))||
           !matrix_valid(bank,&entry->gate,width,hidden)||!matrix_valid(bank,&entry->up,width,hidden)||
           !matrix_valid(bank,&entry->down,hidden,width))goto invalid;
    }
    *out=bank;return 0;
invalid:destroy(bank);return fail(error,error_bytes,"native original expert files or matrix inventory differ");
}
int oa_original_experts_retain(oa_original_experts *bank) {
    uint_fast64_t refs;if(!bank)return -1;refs=atomic_load(&bank->refs);
    do {if(!refs||refs==UINT64_MAX)return -1;}
    while(!atomic_compare_exchange_weak(&bank->refs,&refs,refs+1));return 0;
}
int oa_original_experts_release(oa_original_experts *bank) {
    uint_fast64_t refs;if(!bank)return -1;refs=atomic_load(&bank->refs);
    do {if(!refs)return -1;}
    while(!atomic_compare_exchange_weak(&bank->refs,&refs,refs-1));
    if(refs==1)destroy(bank);return 0;
}
int oa_original_expert_evaluate_cancel(void *owner,int32_t layer,int32_t expert,const float *input,float *output,
    int (*cancel)(void *),void *cancel_data) {
    oa_original_experts *bank=(oa_original_experts *)owner;uint32_t left=0,right,index;uint64_t key,i;int status=1;
    if(!bank||!input||!output||layer<0||expert<0||oa_original_experts_retain(bank))return 1;
    if(atomic_flag_test_and_set(&bank->guard)){oa_original_experts_release(bank);return 1;}
    key=(uint64_t)(uint32_t)layer<<32|(uint32_t)expert;right=bank->expert_count;
    while(left<right) {index=left+(right-left)/2;uint64_t found=(uint64_t)bank->experts[index].layer<<32|bank->experts[index].expert;
        if(found<key)left=index+1;else right=index;}
    if(left==bank->expert_count||bank->experts[left].layer!=(uint32_t)layer||bank->experts[left].expert!=(uint32_t)expert)goto done;
    for(i=0;i<bank->hidden;i++)if(!isfinite(input[i]))goto done;
    if(project(bank,&bank->experts[left].gate,input,bank->gate,cancel,cancel_data,0)||
       project(bank,&bank->experts[left].up,input,bank->up,cancel,cancel_data,0))goto done;
    for(i=0;i<bank->width;i++) {
        float gate=bank->gate[i]<bank->limit?bank->gate[i]:bank->limit;
        float up=bank->up[i]>bank->limit?bank->limit:bank->up[i]<-bank->limit?-bank->limit:bank->up[i];
        float sigmoid=gate>=0?1/(1+expf(-gate)):expf(gate)/(1+expf(gate));
        bank->gate[i]=(gate*sigmoid)*up;if(!isfinite(bank->gate[i]))goto done;
    }
    if(project(bank,&bank->experts[left].down,bank->gate,bank->row,cancel,cancel_data,0))goto done;
    if(cancel&&cancel(cancel_data))goto done;
    memcpy(output,bank->row,(size_t)bank->hidden*4);status=0;
done:atomic_flag_clear(&bank->guard);oa_original_experts_release(bank);return status;
}
int oa_original_expert_evaluate(void *owner,int32_t layer,int32_t expert,const float *input,float *output) {
    return oa_original_expert_evaluate_cancel(owner,layer,expert,input,output,NULL,NULL);
}
int oa_original_experts_supports(oa_original_experts *bank,uint32_t layer,uint32_t expert,uint64_t hidden,uint64_t width) {
    uint32_t left=0,right,index;uint64_t key;
    if(!bank||bank->hidden!=hidden||bank->width!=width)return 0;
    key=(uint64_t)layer<<32|expert;right=bank->expert_count;
    while(left<right) {
        uint64_t found;index=left+(right-left)/2;
        found=(uint64_t)bank->experts[index].layer<<32|bank->experts[index].expert;
        if(found<key)left=index+1;else right=index;
    }
    return left<bank->expert_count&&bank->experts[left].layer==layer&&bank->experts[left].expert==expert;
}
int oa_original_expert_evaluate64(void *owner,uint64_t layer,uint64_t expert,const float *input,float *output) {
    if(layer>INT32_MAX||expert>INT32_MAX)return 1;
    return oa_original_expert_evaluate(owner,(int32_t)layer,(int32_t)expert,input,output);
}
int oa_original_expert_evaluate_counted64(void *owner,uint64_t layer,uint64_t expert,
    const float *input,uint64_t input_count,float *output,uint64_t output_count) {
    oa_original_experts *bank=(oa_original_experts *)owner;int status;
    if(!bank||layer>INT32_MAX||expert>INT32_MAX||oa_original_experts_retain(bank))return 1;
    if(input_count!=bank->hidden||output_count!=bank->hidden){oa_original_experts_release(bank);return 1;}
    status=oa_original_expert_evaluate(owner,(int32_t)layer,(int32_t)expert,input,output);
    oa_original_experts_release(bank);return status;
}
size_t oa_original_experts_layout(uint32_t kind,uint64_t *out,size_t capacity) {
#define FIELD(type,name) offsetof(type,name)
    const uint64_t files[]={sizeof(oa_original_file),3,FIELD(oa_original_file,path),FIELD(oa_original_file,size),FIELD(oa_original_file,mtime_ns)};
    const uint64_t matrices[]={sizeof(oa_original_matrix),14,FIELD(oa_original_matrix,file),FIELD(oa_original_matrix,dtype),
        FIELD(oa_original_matrix,scale_file),FIELD(oa_original_matrix,scale_dtype),FIELD(oa_original_matrix,offset),
        FIELD(oa_original_matrix,elements),FIELD(oa_original_matrix,element_offset),FIELD(oa_original_matrix,rows),
        FIELD(oa_original_matrix,columns),FIELD(oa_original_matrix,parent_rows),FIELD(oa_original_matrix,scale_offset),
        FIELD(oa_original_matrix,scale_elements),FIELD(oa_original_matrix,block_rows),FIELD(oa_original_matrix,block_columns)};
    const uint64_t experts[]={sizeof(oa_original_expert),5,FIELD(oa_original_expert,layer),FIELD(oa_original_expert,expert),
        FIELD(oa_original_expert,gate),FIELD(oa_original_expert,up),FIELD(oa_original_expert,down)};
    const uint64_t *values=NULL;size_t count=0;
    if(kind==1){values=files;count=sizeof(files)/sizeof(*files);}
    else if(kind==2){values=matrices;count=sizeof(matrices)/sizeof(*matrices);}
    else if(kind==3){values=experts;count=sizeof(experts)/sizeof(*experts);}
    if(out&&capacity>=count&&values)memcpy(out,values,count*sizeof(*out));return count;
#undef FIELD
}
