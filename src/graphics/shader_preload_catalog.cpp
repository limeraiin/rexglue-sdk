#include <rex/graphics/shader_preload_catalog.h>
#include <rex/graphics/nr_shader_db.h>
#include <rex/hash.h>
#include <fmt/format.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

namespace rex::graphics {
namespace {
using Bytes = std::vector<uint8_t>;
// Matches ShaderStoredHeader and PipelineStoredDescription in pipeline_cache.h.
// Reject other versions rather than interpreting incompatible records.
const Bytes kShaderHeader{'X','E','S','H',0x20,0x20,0x12,0x19};
const Bytes kPipelineHeader{'X','E','P','S','D','X','R','T',0x20,0x26,0x09,0x22};
constexpr size_t kPipelineRecordBytes = 72;
template<class T> T Read(const Bytes& b, size_t offset) {
  T v;
  if (offset > b.size() || sizeof(T) > b.size()-offset)
    throw std::runtime_error("Truncated shader catalog record");
  std::memcpy(&v,b.data()+offset,sizeof(T));
  return v;
}
template<class T> void Append(Bytes& b, T v) {
  const auto* p = reinterpret_cast<const uint8_t*>(&v);
  b.insert(b.end(),p,p+sizeof(v));
}
Bytes Load(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) return {};
  std::ifstream f(path,std::ios::binary|std::ios::ate);
  const auto size=f.tellg();
  if (!f || size < 0 || size > 256*1024*1024)
    throw std::runtime_error("Cannot read shader catalog: "+path.string());
  Bytes b(static_cast<size_t>(size));
  f.seekg(0);
  if (!b.empty() && !f.read(reinterpret_cast<char*>(b.data()),b.size()))
    throw std::runtime_error("Incomplete shader catalog read: "+path.string());
  return b;
}
void Header(const Bytes& b,const Bytes& expected) {
  if (b.size()<expected.size() || !std::equal(expected.begin(),expected.end(),b.begin()))
    throw std::runtime_error("Incompatible shader catalog format");
}
void Save(const std::filesystem::path& path,const Bytes& b) {
  if (Load(path)==b) return;
  std::filesystem::create_directories(path.parent_path());
  auto temp=path; temp += ".preload.tmp";
  {
    std::ofstream f(temp,std::ios::binary|std::ios::trunc);
    f.write(reinterpret_cast<const char*>(b.data()),b.size());
    f.flush();
    if (!f) throw std::runtime_error("Cannot write shader catalog: "+path.string());
  }
#ifdef _WIN32
  if (!MoveFileExW(temp.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
    throw std::runtime_error("Cannot install shader catalog: "+path.string());
#else
  std::filesystem::rename(temp,path);
#endif
}
using Shaders=std::map<uint64_t,Bytes>;
void InsertShader(Shaders& shaders,uint64_t hash,Bytes record) {
  auto [it,added]=shaders.emplace(hash,record);
  if (!added && it->second!=record)
    throw std::runtime_error("Conflicting shader records");
}
void ReadShaders(const Bytes& b,Shaders& out) {
  if (b.empty()) return;
  Header(b,kShaderHeader);
  for (size_t p=kShaderHeader.size();p<b.size();) {
    const uint64_t hash=Read<uint64_t>(b,p);
    const size_t bytes=size_t(Read<uint32_t>(b,p+8)&0x7FFFFFFFu)*4;
    if (!bytes || p+12>b.size() || bytes>b.size()-(p+12) ||
        XXH3_64bits(b.data()+p+12,bytes)!=hash)
      throw std::runtime_error("Invalid shader catalog hash or bounds");
    InsertShader(out,hash,Bytes(b.begin()+p,b.begin()+p+12+bytes));
    p+=12+bytes;
  }
}
void ReadPipelines(const Bytes& b,std::set<Bytes>& out) {
  if (b.empty()) return;
  Header(b,kPipelineHeader);
  if ((b.size()-kPipelineHeader.size())%kPipelineRecordBytes)
    throw std::runtime_error("Truncated pipeline catalog");
  for (size_t p=kPipelineHeader.size();p<b.size();p+=kPipelineRecordBytes) {
    if (Read<uint64_t>(b,p)!=XXH3_64bits(b.data()+p+8,kPipelineRecordBytes-8))
      throw std::runtime_error("Invalid pipeline catalog hash");
    out.emplace(b.begin()+p,b.begin()+p+kPipelineRecordBytes);
  }
}
}

ShaderPreloadCatalogResult PrepareShaderPreloadCatalog(
    const std::filesystem::path& shareable,const std::filesystem::path& catalog,
    const std::filesystem::path& database,uint32_t title_id) {
  const auto stem=fmt::format("{:08X}",title_id);
  const auto shader_name=stem+".xsh";
  const auto pipeline_name=stem+".rtv.d3d12.xpso";
  Shaders shaders;
  std::set<Bytes> pipelines;
  ReadShaders(Load(shareable/shader_name),shaders);
  ReadPipelines(Load(shareable/pipeline_name),pipelines);
  if (!catalog.empty()) {
    ReadShaders(Load(catalog/shader_name),shaders);
    ReadPipelines(Load(catalog/pipeline_name),pipelines);
  }
  ShaderPreloadCatalogResult result;
  if (!database.empty()) {
    nr::ShaderDbIndex db;
    // Read using filesystem::path so Unicode installation paths work on Windows.
    if (!db.LoadBytes(Load(database)) || !db.unique_count() ||
        db.stats().bad_ucode_bounds || db.stats().stage_token_mismatch)
      throw std::runtime_error("Missing or invalid game shader database: "+database.string());
    result.database_shaders=uint32_t(db.unique_count());
    for (const auto& c:db.containers()) {
      Bytes record;
      Append(record,c.ucode_hash);
      Append(record,uint32_t(c.ucode_bytes/4)|(c.pixel?0x80000000u:0u));
      record.insert(record.end(),c.ucode,c.ucode+c.ucode_bytes);
      InsertShader(shaders,c.ucode_hash,std::move(record));
    }
  }
  for (const auto& p:pipelines) {
    const auto vs=Read<uint64_t>(p,8), ps=Read<uint64_t>(p,24);
    auto valid_stage=[&](uint64_t hash,bool pixel) {
      auto it=shaders.find(hash);
      return it!=shaders.end() && bool(Read<uint32_t>(it->second,8)>>31)==pixel;
    };
    if (!valid_stage(vs,false) || (ps && !valid_stage(ps,true)))
      throw std::runtime_error("Pipeline catalog references a missing shader or wrong stage");
  }
  Bytes shader_bytes=kShaderHeader,pipeline_bytes=kPipelineHeader;
  for (const auto& [hash,record]:shaders)
    shader_bytes.insert(shader_bytes.end(),record.begin(),record.end());
  for (const auto& record:pipelines)
    pipeline_bytes.insert(pipeline_bytes.end(),record.begin(),record.end());
  Save(shareable/shader_name,shader_bytes);
  Save(shareable/pipeline_name,pipeline_bytes);
  result.shaders=uint32_t(shaders.size());
  result.pipelines=uint32_t(pipelines.size());
  return result;
}
}
