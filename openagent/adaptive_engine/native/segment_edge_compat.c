#include "segment_edge_compat.h"
#include <stdatomic.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Registry mutation is serialized; adapter calls execute outside that lock.
 * A handle permits one active operation. Independent sessions may run in
 * parallel. Adapter code must retain immutable engine weights for its children
 * and implement its advertised numeric/cache contract. Void destroy calls must
 * not race an operation on the same handle; busy handles are left alive. */
#define SE_ADAPTERS 64u
#define SE_HANDLES 65536u
#define SE_BACKENDS UINT64_C(0x1f00)
#define SE_SEGMENT_FLAGS (SE_BACKENDS | UINT64_C(15))
#define SE_EDGE_FLAGS (SE_BACKENDS | UINT64_C(15))
enum { SE_ENGINE=1, SE_SESSION=2, SE_EDGE=3 };
typedef struct se_object se_object;
struct se_object {
    se_object *next, *parent;
    void *implementation;
    unsigned kind, busy, children;
    uint32_t context;
    uint64_t position;
    const oa_se_segment_adapter *segment;
    const oa_se_edge_adapter *edge;
    oa_se_segment_caps segment_caps;
    oa_se_edge_caps edge_caps;
};
static atomic_flag se_guard = ATOMIC_FLAG_INIT;
static oa_se_segment_adapter segment_adapters[SE_ADAPTERS];
static oa_se_edge_adapter edge_adapters[SE_ADAPTERS];
static char segment_names[SE_ADAPTERS][64], edge_names[SE_ADAPTERS][64];
static unsigned segment_count, edge_count, segment_sealed, edge_sealed, object_count;
static se_object *objects;

#define SE_FIELD(type,name) ((uint64_t)offsetof(type,name))
size_t oa_segment_edge_layout(uint32_t record,uint64_t *values,size_t capacity) {
    const uint64_t *layout=NULL;size_t count=0;
    const uint64_t segment_caps[]={sizeof(oa_se_segment_caps),13,
        SE_FIELD(oa_se_segment_caps,size),SE_FIELD(oa_se_segment_caps,version),SE_FIELD(oa_se_segment_caps,flags),
        SE_FIELD(oa_se_segment_caps,engine),SE_FIELD(oa_se_segment_caps,schema),SE_FIELD(oa_se_segment_caps,numeric),
        SE_FIELD(oa_se_segment_caps,dtype),SE_FIELD(oa_se_segment_caps,width),SE_FIELD(oa_se_segment_caps,rows),
        SE_FIELD(oa_se_segment_caps,context),SE_FIELD(oa_se_segment_caps,layers),
        SE_FIELD(oa_se_segment_caps,reserved32),SE_FIELD(oa_se_segment_caps,reserved64)};
    const uint64_t edge_caps[]={sizeof(oa_se_edge_caps),18,
        SE_FIELD(oa_se_edge_caps,size),SE_FIELD(oa_se_edge_caps,version),SE_FIELD(oa_se_edge_caps,flags),
        SE_FIELD(oa_se_edge_caps,engine),SE_FIELD(oa_se_edge_caps,schema),SE_FIELD(oa_se_edge_caps,numeric),
        SE_FIELD(oa_se_edge_caps,tokenizer),SE_FIELD(oa_se_edge_caps,dtype),SE_FIELD(oa_se_edge_caps,width),
        SE_FIELD(oa_se_edge_caps,vocab),SE_FIELD(oa_se_edge_caps,rows),SE_FIELD(oa_se_edge_caps,context),
        SE_FIELD(oa_se_edge_caps,layers),SE_FIELD(oa_se_edge_caps,bos),SE_FIELD(oa_se_edge_caps,eos),
        SE_FIELD(oa_se_edge_caps,reserved32),SE_FIELD(oa_se_edge_caps,resident_bytes),SE_FIELD(oa_se_edge_caps,reserved64)};
    const uint64_t segment_options[]={sizeof(oa_se_segment_options),10,
        SE_FIELD(oa_se_segment_options,size),SE_FIELD(oa_se_segment_options,model),
        SE_FIELD(oa_se_segment_options,begin),SE_FIELD(oa_se_segment_options,end),
        SE_FIELD(oa_se_segment_options,context),SE_FIELD(oa_se_segment_options,reserved),
        SE_FIELD(oa_se_segment_options,memory),SE_FIELD(oa_se_segment_options,backend),
        SE_FIELD(oa_se_segment_options,plan),SE_FIELD(oa_se_segment_options,plan_size)};
    const uint64_t session_options[]={sizeof(oa_se_session_options),4,
        SE_FIELD(oa_se_session_options,size),SE_FIELD(oa_se_session_options,context),
        SE_FIELD(oa_se_session_options,memory),SE_FIELD(oa_se_session_options,reserved)};
    const uint64_t edge_options[]={sizeof(oa_se_edge_options),7,
        SE_FIELD(oa_se_edge_options,size),SE_FIELD(oa_se_edge_options,model),
        SE_FIELD(oa_se_edge_options,memory),SE_FIELD(oa_se_edge_options,backend),
        SE_FIELD(oa_se_edge_options,plan),SE_FIELD(oa_se_edge_options,plan_size),SE_FIELD(oa_se_edge_options,reserved)};
    const uint64_t run[]={sizeof(oa_se_run),12,
        SE_FIELD(oa_se_run,size),SE_FIELD(oa_se_run,rows),SE_FIELD(oa_se_run,position),
        SE_FIELD(oa_se_run,tokens),SE_FIELD(oa_se_run,token_count),SE_FIELD(oa_se_run,input),
        SE_FIELD(oa_se_run,input_bytes),SE_FIELD(oa_se_run,output),SE_FIELD(oa_se_run,output_bytes),
        SE_FIELD(oa_se_run,cancel),SE_FIELD(oa_se_run,cancel_data),SE_FIELD(oa_se_run,reserved)};
    const uint64_t embed[]={sizeof(oa_se_embed),9,
        SE_FIELD(oa_se_embed,size),SE_FIELD(oa_se_embed,rows),SE_FIELD(oa_se_embed,tokens),
        SE_FIELD(oa_se_embed,token_count),SE_FIELD(oa_se_embed,output),SE_FIELD(oa_se_embed,output_bytes),
        SE_FIELD(oa_se_embed,cancel),SE_FIELD(oa_se_embed,cancel_data),SE_FIELD(oa_se_embed,reserved)};
    const uint64_t logits[]={sizeof(oa_se_logits),9,
        SE_FIELD(oa_se_logits,size),SE_FIELD(oa_se_logits,rows),SE_FIELD(oa_se_logits,input),
        SE_FIELD(oa_se_logits,input_bytes),SE_FIELD(oa_se_logits,logits),SE_FIELD(oa_se_logits,logits_capacity),
        SE_FIELD(oa_se_logits,cancel),SE_FIELD(oa_se_logits,cancel_data),SE_FIELD(oa_se_logits,reserved)};
    const uint64_t segment_adapter[]={sizeof(oa_se_segment_adapter),11,
        SE_FIELD(oa_se_segment_adapter,size),SE_FIELD(oa_se_segment_adapter,version),SE_FIELD(oa_se_segment_adapter,engine),
        SE_FIELD(oa_se_segment_adapter,open),SE_FIELD(oa_se_segment_adapter,destroy),SE_FIELD(oa_se_segment_adapter,create),
        SE_FIELD(oa_se_segment_adapter,destroy_session),SE_FIELD(oa_se_segment_adapter,run),
        SE_FIELD(oa_se_segment_adapter,snapshot),SE_FIELD(oa_se_segment_adapter,restore),SE_FIELD(oa_se_segment_adapter,reserved)};
    const uint64_t edge_adapter[]={sizeof(oa_se_edge_adapter),11,
        SE_FIELD(oa_se_edge_adapter,size),SE_FIELD(oa_se_edge_adapter,version),SE_FIELD(oa_se_edge_adapter,engine),
        SE_FIELD(oa_se_edge_adapter,open),SE_FIELD(oa_se_edge_adapter,destroy),SE_FIELD(oa_se_edge_adapter,tokenize),
        SE_FIELD(oa_se_edge_adapter,detokenize),SE_FIELD(oa_se_edge_adapter,embed),SE_FIELD(oa_se_edge_adapter,select),
        SE_FIELD(oa_se_edge_adapter,logits),SE_FIELD(oa_se_edge_adapter,reserved)};
    const uint64_t select[]={sizeof(oa_se_select),11,
        SE_FIELD(oa_se_select,size),SE_FIELD(oa_se_select,rows),SE_FIELD(oa_se_select,input),
        SE_FIELD(oa_se_select,input_bytes),SE_FIELD(oa_se_select,tokens),SE_FIELD(oa_se_select,token_capacity),
        SE_FIELD(oa_se_select,scores),SE_FIELD(oa_se_select,score_capacity),SE_FIELD(oa_se_select,cancel),
        SE_FIELD(oa_se_select,cancel_data),SE_FIELD(oa_se_select,reserved)};
    switch(record) {
    case 1:layout=segment_caps;count=sizeof(segment_caps)/sizeof(*segment_caps);break;
    case 2:layout=edge_caps;count=sizeof(edge_caps)/sizeof(*edge_caps);break;
    case 3:layout=segment_options;count=sizeof(segment_options)/sizeof(*segment_options);break;
    case 4:layout=session_options;count=sizeof(session_options)/sizeof(*session_options);break;
    case 5:layout=edge_options;count=sizeof(edge_options)/sizeof(*edge_options);break;
    case 6:layout=run;count=sizeof(run)/sizeof(*run);break;
    case 7:layout=embed;count=sizeof(embed)/sizeof(*embed);break;
    case 8:layout=logits;count=sizeof(logits)/sizeof(*logits);break;
    case 9:layout=segment_adapter;count=sizeof(segment_adapter)/sizeof(*segment_adapter);break;
    case 10:layout=edge_adapter;count=sizeof(edge_adapter)/sizeof(*edge_adapter);break;
    case 11:layout=select;count=sizeof(select)/sizeof(*select);break;
    default:return 0;
    }
    if(!values || capacity<count) return 0;
    memcpy(values,layout,count*sizeof(*values));return count;
}
#undef SE_FIELD

static void lock_registry(void) {
    while (atomic_flag_test_and_set_explicit(&se_guard, memory_order_acquire)) {}
}
static void unlock_registry(void) {
    atomic_flag_clear_explicit(&se_guard, memory_order_release);
}
static int failure(char *error, size_t capacity, const char *message) {
    if (error && capacity) {
        size_t n=strlen(message);
        if (n>=capacity) n=capacity-1;
        memcpy(error,message,n); error[n]=0;
    }
    return -1;
}
static int name_ok(const char *name, size_t capacity) {
    size_t i;
    if (!name || !name[0]) return 0;
    for (i=0;i<capacity;i++) if (!name[i]) return 1;
    return 0;
}
static int zero64(const uint64_t *values, size_t count) {
    size_t i;
    for(i=0;i<count;i++) if(values[i]) return 0;
    return 1;
}
static int zero32(const uint32_t *values, size_t count) {
    size_t i;
    for(i=0;i<count;i++) if(values[i]) return 0;
    return 1;
}
static int bytes_for(uint32_t rows, uint32_t width, uint32_t dtype, size_t *bytes) {
    size_t unit=dtype==1?4:(dtype==2 || dtype==3?2:0);
    if (!rows || !width || !unit || (size_t)rows>SIZE_MAX/(size_t)width) return 0;
    *bytes=(size_t)rows*width;
    if (*bytes>SIZE_MAX/unit) return 0;
    *bytes*=unit; return 1;
}
static int regions_overlap(const void *a, size_t na, const void *b, size_t nb) {
    uintptr_t x=(uintptr_t)a, y=(uintptr_t)b;
    if (!na || !nb) return 0;
    if (!a || !b || na>UINTPTR_MAX-x || nb>UINTPTR_MAX-y) return 1;
    return x<y+nb && y<x+na;
}
static se_object *find_object(const void *handle) {
    se_object *node;
    for(node=objects;node;node=node->next) if(node==handle) return node;
    return NULL;
}
static se_object *acquire(void *handle, unsigned kind, char *error, size_t size) {
    se_object *node;
    lock_registry(); node=find_object(handle);
    if (!node || node->kind!=kind || node->busy) {
        unlock_registry(); failure(error,size,"invalid or busy native handle"); return NULL;
    }
    node->busy=1; unlock_registry(); return node;
}
static void release(se_object *node) {
    lock_registry(); node->busy=0; unlock_registry();
}
static int publish(se_object *node, char *error, size_t size) {
    lock_registry();
    if(object_count>=SE_HANDLES) {
        unlock_registry(); return failure(error,size,"native handle inventory exhausted");
    }
    node->next=objects; objects=node; ++object_count;
    if(node->parent) ++node->parent->children;
    unlock_registry(); return 0;
}
static void remove_object(se_object *node) {
    se_object **cursor;
    lock_registry();
    for(cursor=&objects;*cursor && *cursor!=node;cursor=&(*cursor)->next) {}
    if(*cursor) {
        *cursor=node->next; --object_count;
        if(node->parent) --node->parent->children;
    }
    unlock_registry(); free(node);
}

int coli_segment_adapter_register(const oa_se_segment_adapter *adapter) {
    unsigned i;
    if(!adapter || adapter->size<sizeof(*adapter) || adapter->version!=1 ||
       !name_ok(adapter->engine,64) || !adapter->open || !adapter->destroy ||
       !adapter->create || !adapter->destroy_session || !adapter->run) return -1;
    for(i=0;i<8;i++) if(adapter->reserved[i]) return -1;
    lock_registry();
    if(segment_sealed || segment_count>=SE_ADAPTERS) { unlock_registry(); return -1; }
    for(i=0;i<segment_count;i++) if(!strcmp(adapter->engine,segment_names[i])) {
        unlock_registry(); return -1;
    }
    i=segment_count++; segment_adapters[i]=*adapter;
    strcpy(segment_names[i],adapter->engine); segment_adapters[i].engine=segment_names[i];
    unlock_registry(); return 0;
}
const oa_se_segment_adapter *coli_segment_adapter_lookup(const char *name) {
    unsigned i; const oa_se_segment_adapter *result=NULL;
    if(!name_ok(name,64)) return NULL;
    lock_registry(); segment_sealed=1;
    for(i=0;i<segment_count;i++) if(!strcmp(name,segment_names[i])) {result=&segment_adapters[i];break;}
    unlock_registry(); return result;
}
int coli_segment_adapter_count(void) {
    int result; lock_registry(); result=(int)segment_count; unlock_registry(); return result;
}
static int segment_caps_ok(const oa_se_segment_caps *caps, const oa_se_segment_adapter *adapter) {
    return caps->size==sizeof(*caps) && caps->version==1 &&
        name_ok(caps->engine,sizeof(caps->engine)) && !strcmp(caps->engine,adapter->engine) &&
        name_ok(caps->schema,sizeof(caps->schema)) && name_ok(caps->numeric,sizeof(caps->numeric)) &&
        caps->dtype>=1 && caps->dtype<=3 && caps->width && caps->rows && caps->context && caps->layers &&
        (caps->flags&SE_BACKENDS) && !(caps->flags&~SE_SEGMENT_FLAGS) &&
        zero32(caps->reserved32,7) && zero64(caps->reserved64,4) &&
        (!(caps->flags&2) || (adapter->snapshot && adapter->restore));
}
int coli_segment_engine_open(const char *name, const oa_se_segment_options *options,
                            void **handle, char *error, size_t size) {
    const oa_se_segment_adapter *adapter; se_object *node; int status;
    if(!handle) return failure(error,size,"engine output required");
    *handle=NULL;
    if(!options || options->size<sizeof(*options) || !options->model || !options->model[0] ||
       options->begin>=options->end || !options->context || options->reserved || options->backend&~SE_BACKENDS ||
       (!options->plan && options->plan_size)) return failure(error,size,"invalid segment options");
    adapter=coli_segment_adapter_lookup(name);
    if(!adapter) return failure(error,size,"segment adapter is not registered");
    node=(se_object*)calloc(1,sizeof(*node));
    if(!node) return failure(error,size,"engine allocation failed");
    node->kind=SE_ENGINE;node->segment=adapter;node->segment_caps.size=sizeof(node->segment_caps);
    status=adapter->open(&node->implementation,&node->segment_caps,options,error,size);
    if(status || !node->implementation || !segment_caps_ok(&node->segment_caps,adapter) ||
       options->end>node->segment_caps.layers || options->context>node->segment_caps.context ||
       (options->backend && !(options->backend&node->segment_caps.flags&SE_BACKENDS))) {
        if(node->implementation) adapter->destroy(node->implementation);
        free(node); return status?status:failure(error,size,"invalid segment capability contract");
    }
    node->context=options->context?options->context:node->segment_caps.context;
    status=publish(node,error,size);
    if(status) {adapter->destroy(node->implementation);free(node);return status;}
    *handle=node;return 0;
}
int coli_segment_engine_capabilities(const void *handle, oa_se_segment_caps *caps,
                                    char *error, size_t size) {
    se_object *node;
    if(!caps || caps->size<sizeof(*caps)) return failure(error,size,"capability output is too small");
    node=acquire((void*)handle,SE_ENGINE,error,size); if(!node) return -1;
    /* `size` is caller-supplied capacity, not proof of the backing allocation.
     * Only touch the known v1 record; zeroing the claimed larger capacity can
     * overwrite unrelated caller memory when a malformed size is supplied. */
    memset(caps,0,sizeof(*caps));memcpy(caps,&node->segment_caps,sizeof(*caps));release(node);return 0;
}
int coli_segment_engine_close(void *handle, char *error, size_t size) {
    se_object *node;
    unsigned children;
    if(!handle) return 0;
    node=acquire(handle,SE_ENGINE,error,size);
    if(!node) return -1;
    lock_registry();children=node->children;unlock_registry();
    if(children) {release(node);return failure(error,size,"segment sessions remain alive");}
    node->segment->destroy(node->implementation);remove_object(node);return 0;
}
int coli_segment_session_create(void *handle, const oa_se_session_options *options,
                               void **session, char *error, size_t size) {
    se_object *engine, *node; int status;unsigned children;
    if(!session) return failure(error,size,"session output required");
    *session=NULL;
    if(!options || options->size<sizeof(*options) || !options->context || !zero64(options->reserved,4))
        return failure(error,size,"invalid session options");
    engine=acquire(handle,SE_ENGINE,error,size);if(!engine) return -1;
    lock_registry();children=engine->children;unlock_registry();
    if(options->context>engine->context ||
       (children && !(engine->segment_caps.flags&8))) {
        release(engine);return failure(error,size,"session context or isolation capability mismatch");
    }
    node=(se_object*)calloc(1,sizeof(*node));
    if(!node) {release(engine);return failure(error,size,"session allocation failed");}
    node->kind=SE_SESSION;node->parent=engine;node->segment=engine->segment;
    node->context=options->context?options->context:engine->context;
    status=node->segment->create(engine->implementation,&node->implementation,options,error,size);
    if(!status && !node->implementation) status=failure(error,size,"adapter returned no session");
    if(!status) status=publish(node,error,size);
    if(status) {
        if(node->implementation) node->segment->destroy_session(node->implementation);
        free(node);release(engine);return status;
    }
    release(engine);*session=node;return 0;
}
void coli_segment_session_destroy(void *handle) {
    se_object *node=acquire(handle,SE_SESSION,NULL,0);if(!node) return;
    node->segment->destroy_session(node->implementation);remove_object(node);
}
int coli_segment_run(void *handle, const oa_se_run *request, char *error, size_t size) {
    se_object *node; const oa_se_segment_caps *caps;size_t bytes;int status;
    if(!request || request->size<sizeof(*request) || !zero64(request->reserved,4))
        return failure(error,size,"invalid segment run request");
    node=acquire(handle,SE_SESSION,error,size);if(!node) return -1;
    caps=&node->parent->segment_caps;
    if(!bytes_for(request->rows,caps->width,caps->dtype,&bytes) || request->rows>caps->rows ||
       request->position>node->context || request->rows>node->context-request->position ||
       !request->input || !request->output || request->input_bytes!=bytes || request->output_bytes!=bytes ||
       regions_overlap(request->input,bytes,request->output,bytes) ||
       (request->tokens?(request->token_count!=request->rows):request->token_count!=0) ||
       ((caps->flags&1) && !request->tokens)) {
        release(node);return failure(error,size,"segment shape, context or token contract mismatch");
    }
    if(request->cancel && request->cancel(request->cancel_data)) {
        release(node);return failure(error,size,"segment run cancelled before dispatch");
    }
    status=node->segment->run(node->implementation,request,error,size);
    if(!status) node->position=request->position+request->rows;
    release(node);return status;
}
int coli_segment_snapshot(void *handle, oa_se_write write, void *user, char *error, size_t size) {
    se_object *node;int status;
    if(!write) return failure(error,size,"snapshot writer required");
    node=acquire(handle,SE_SESSION,error,size);if(!node) return -1;
    if(!(node->parent->segment_caps.flags&2) || !node->segment->snapshot) {
        release(node);return failure(error,size,"snapshot capability unavailable");
    }
    status=node->segment->snapshot(node->implementation,write,user,error,size);release(node);return status;
}
int coli_segment_restore(void *handle, oa_se_read read, void *user, char *error, size_t size) {
    se_object *node;int status;
    if(!read) return failure(error,size,"snapshot reader required");
    node=acquire(handle,SE_SESSION,error,size);if(!node) return -1;
    if(!(node->parent->segment_caps.flags&2) || !node->segment->restore) {
        release(node);return failure(error,size,"restore capability unavailable");
    }
    status=node->segment->restore(node->implementation,read,user,error,size);release(node);return status;
}

int coli_edge_adapter_register(const oa_se_edge_adapter *adapter) {
    unsigned i;
    if(!adapter || adapter->size<sizeof(*adapter) || adapter->version!=2 ||
       !name_ok(adapter->engine,64) || !adapter->open || !adapter->destroy || !adapter->embed) return -1;
    for(i=0;i<7;i++) if(adapter->reserved[i]) return -1;
    lock_registry();
    if(edge_sealed || edge_count>=SE_ADAPTERS) {unlock_registry();return -1;}
    for(i=0;i<edge_count;i++) if(!strcmp(adapter->engine,edge_names[i])) {unlock_registry();return -1;}
    i=edge_count++;edge_adapters[i]=*adapter;
    strcpy(edge_names[i],adapter->engine);edge_adapters[i].engine=edge_names[i];
    unlock_registry();return 0;
}
const oa_se_edge_adapter *coli_edge_adapter_lookup(const char *name) {
    unsigned i;const oa_se_edge_adapter *result=NULL;
    if(!name_ok(name,64)) return NULL;
    lock_registry();edge_sealed=1;
    for(i=0;i<edge_count;i++) if(!strcmp(name,edge_names[i])) {result=&edge_adapters[i];break;}
    unlock_registry();return result;
}
int coli_edge_adapter_count(void) {
    int result;lock_registry();result=(int)edge_count;unlock_registry();return result;
}
static int edge_caps_ok(const oa_se_edge_caps *caps,const oa_se_edge_adapter *adapter) {
    return caps->size==sizeof(*caps) && caps->version==2 &&
        name_ok(caps->engine,sizeof(caps->engine)) && !strcmp(caps->engine,adapter->engine) &&
        name_ok(caps->schema,sizeof(caps->schema)) && name_ok(caps->numeric,sizeof(caps->numeric)) &&
        caps->dtype>=1 && caps->dtype<=3 && caps->width && caps->rows && caps->context && caps->layers &&
        caps->vocab && caps->vocab<=INT32_MAX && (caps->flags&SE_BACKENDS) &&
        !(caps->flags&~SE_EDGE_FLAGS) && zero32(caps->reserved32,4) && zero64(caps->reserved64,3) &&
        (!(caps->flags&1) || (adapter->tokenize && name_ok(caps->tokenizer,sizeof(caps->tokenizer)))) &&
        (!(caps->flags&2) || adapter->detokenize) && (!(caps->flags&4) || adapter->select) &&
        (!(caps->flags&8) || adapter->logits) && caps->bos>=-1 && caps->eos>=-1 &&
        (caps->bos<0 || (uint32_t)caps->bos<caps->vocab) && (caps->eos<0 || (uint32_t)caps->eos<caps->vocab);
}
int coli_edge_engine_open(const char *name,const oa_se_edge_options *options,void **handle,char *error,size_t size) {
    const oa_se_edge_adapter *adapter;se_object *node;int status;
    if(!handle) return failure(error,size,"edge output required");
    *handle=NULL;
    if(!options || options->size<sizeof(*options) || !options->model || !options->model[0] ||
       options->backend&~SE_BACKENDS || (!options->plan && options->plan_size) || !zero64(options->reserved,3))
        return failure(error,size,"invalid edge options");
    adapter=coli_edge_adapter_lookup(name);if(!adapter) return failure(error,size,"edge adapter is not registered");
    node=(se_object*)calloc(1,sizeof(*node));if(!node) return failure(error,size,"edge allocation failed");
    node->kind=SE_EDGE;node->edge=adapter;node->edge_caps.size=sizeof(node->edge_caps);
    status=adapter->open(&node->implementation,&node->edge_caps,options,error,size);
    if(status || !node->implementation || !edge_caps_ok(&node->edge_caps,adapter) ||
       (options->backend && !(options->backend&node->edge_caps.flags&SE_BACKENDS)) ||
       (options->memory && node->edge_caps.resident_bytes>options->memory)) {
        if(node->implementation) adapter->destroy(node->implementation);
        free(node);return status?status:failure(error,size,"invalid edge capability contract");
    }
    status=publish(node,error,size);
    if(status) {adapter->destroy(node->implementation);free(node);return status;}
    *handle=node;return 0;
}
int coli_edge_engine_capabilities(const void *handle,oa_se_edge_caps *caps,char *error,size_t size) {
    se_object *node;
    if(!caps || caps->size<sizeof(*caps)) return failure(error,size,"edge capability output too small");
    node=acquire((void*)handle,SE_EDGE,error,size);if(!node) return -1;
    /* As above, never trust a caller-declared capacity as an actual allocation
     * extent when clearing a fixed-version capability record. */
    memset(caps,0,sizeof(*caps));memcpy(caps,&node->edge_caps,sizeof(*caps));release(node);return 0;
}
void coli_edge_engine_close(void *handle) {
    se_object *node=acquire(handle,SE_EDGE,NULL,0);if(!node) return;
    node->edge->destroy(node->implementation);remove_object(node);
}
int coli_edge_tokenize(void *handle,const char *text,size_t bytes,int32_t *tokens,size_t capacity,
                       size_t *count,char *error,size_t size) {
    se_object *node;int status;size_t i;
    if(!count || !text || (!!tokens != !!capacity) || capacity>SIZE_MAX/sizeof(*tokens))
        return failure(error,size,"invalid tokenizer buffers");
    *count=0;node=acquire(handle,SE_EDGE,error,size);if(!node) return -1;
    if(!(node->edge_caps.flags&1)) {release(node);return failure(error,size,"tokenizer capability unavailable");}
    status=node->edge->tokenize(node->implementation,text,bytes,tokens,capacity,count,error,size);
    if(!status && tokens) {
        if(*count>capacity) status=failure(error,size,"tokenizer returned an invalid extent");
        else for(i=0;i<*count;i++) if(tokens[i]<0 || (uint32_t)tokens[i]>=node->edge_caps.vocab) {
            status=failure(error,size,"tokenizer returned an invalid token");break;
        }
    }
    release(node);return status;
}
int coli_edge_detokenize(void *handle,const int32_t *tokens,size_t count,char *text,size_t capacity,
                         size_t *bytes,char *error,size_t size) {
    se_object *node;int status;size_t i;
    if(!bytes || !tokens || !count || (!!text != !!capacity)) return failure(error,size,"invalid detokenizer buffers");
    *bytes=0;node=acquire(handle,SE_EDGE,error,size);if(!node) return -1;
    if(!(node->edge_caps.flags&2)) {release(node);return failure(error,size,"detokenizer capability unavailable");}
    for(i=0;i<count;i++) if(tokens[i]<0 || (uint32_t)tokens[i]>=node->edge_caps.vocab) {
        release(node);return failure(error,size,"invalid detokenizer token");
    }
    status=node->edge->detokenize(node->implementation,tokens,count,text,capacity,bytes,error,size);
    if(!status && text && *bytes>capacity) status=failure(error,size,"detokenizer returned an invalid extent");
    release(node);return status;
}
int coli_edge_embed(void *handle,const oa_se_embed *request,char *error,size_t size) {
    se_object *node;size_t bytes,i;int status;
    if(!request || request->size<sizeof(*request) || !zero64(request->reserved,3))
        return failure(error,size,"invalid embedding request");
    node=acquire(handle,SE_EDGE,error,size);if(!node) return -1;
    if(!bytes_for(request->rows,node->edge_caps.width,node->edge_caps.dtype,&bytes) ||
       request->rows>node->edge_caps.rows || !request->tokens || request->token_count!=request->rows ||
       !request->output || request->output_bytes!=bytes ||
       request->token_count>SIZE_MAX/sizeof(int32_t) ||
       regions_overlap(request->tokens,request->token_count*sizeof(int32_t),request->output,bytes)) {
        release(node);return failure(error,size,"embedding shape mismatch");
    }
    for(i=0;i<request->token_count;i++) if(request->tokens[i]<0 || (uint32_t)request->tokens[i]>=node->edge_caps.vocab) {
        release(node);return failure(error,size,"embedding token out of range");
    }
    if(request->cancel && request->cancel(request->cancel_data)) {
        release(node);return failure(error,size,"embedding cancelled");
    }
    status=node->edge->embed(node->implementation,request,error,size);release(node);return status;
}
int coli_edge_select(void *handle,const oa_se_select *request,char *error,size_t size) {
    se_object *node;size_t bytes,i;int status;
    if(!request || request->size<sizeof(*request) || !zero64(request->reserved,3))
        return failure(error,size,"invalid selection request");
    node=acquire(handle,SE_EDGE,error,size);if(!node) return -1;
    if(!(node->edge_caps.flags&4) || !bytes_for(request->rows,node->edge_caps.width,node->edge_caps.dtype,&bytes) ||
       request->rows>node->edge_caps.rows || !request->input || request->input_bytes!=bytes ||
       !request->tokens || request->token_capacity<request->rows ||
       (request->scores?request->score_capacity<request->rows:request->score_capacity!=0) ||
       request->rows>SIZE_MAX/sizeof(int32_t) ||
       regions_overlap(request->input,bytes,request->tokens,(size_t)request->rows*sizeof(int32_t)) ||
       (request->scores && (regions_overlap(request->input,bytes,request->scores,(size_t)request->rows*sizeof(float)) ||
        regions_overlap(request->tokens,(size_t)request->rows*sizeof(int32_t),request->scores,(size_t)request->rows*sizeof(float))))) {
        release(node);return failure(error,size,"selection shape or capability mismatch");
    }
    if(request->cancel && request->cancel(request->cancel_data)) {
        release(node);return failure(error,size,"selection cancelled");
    }
    status=node->edge->select(node->implementation,request,error,size);
    if(!status) for(i=0;i<request->rows;i++) {
        if(request->tokens[i]<0 || (uint32_t)request->tokens[i]>=node->edge_caps.vocab ||
           (request->scores && !isfinite(request->scores[i]))) {
            status=failure(error,size,"selection returned invalid token IDs or scores");break;
        }
    }
    release(node);return status;
}
int coli_edge_logits(void *handle,const oa_se_logits *request,char *error,size_t size) {
    se_object *node;size_t bytes,count;int status;
    if(!request || request->size<sizeof(*request) || !zero64(request->reserved,3))
        return failure(error,size,"invalid logits request");
    node=acquire(handle,SE_EDGE,error,size);if(!node) return -1;
    if(!(node->edge_caps.flags&8) || !bytes_for(request->rows,node->edge_caps.width,node->edge_caps.dtype,&bytes) ||
       request->rows>node->edge_caps.rows || !request->input || request->input_bytes!=bytes ||
       !request->logits || (size_t)request->rows>SIZE_MAX/node->edge_caps.vocab) {
        release(node);return failure(error,size,"logits shape or capability mismatch");
    }
    count=(size_t)request->rows*node->edge_caps.vocab;
    if(count>SIZE_MAX/sizeof(float) || request->logits_capacity<count ||
       regions_overlap(request->input,bytes,request->logits,count*sizeof(float))) {
        release(node);return failure(error,size,"logits extent or overlap mismatch");
    }
    if(request->cancel && request->cancel(request->cancel_data)) {
        release(node);return failure(error,size,"logits cancelled");
    }
    status=node->edge->logits(node->implementation,request,error,size);release(node);return status;
}
