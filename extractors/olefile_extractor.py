# extractors/olefile_extractor.py
# Извлечение текста из .doc файлов через olefile (OLE2 binary parser)

from pathlib import Path
import struct
from typing import Optional, Dict

from config import cfg
from logger_utils import logger, assess_text_quality


def extract_doc_olefile(file_path: Path) -> Dict:
    """Извлечение текста из .doc файла через olefile."""
    
    base_result = {
        "source": str(file_path),
        "type": "doc",
        "text": None,
        "image": None,
        "error": None,
        "text_quality": "none"
    }
    
    # Fallback на olefile если Pandoc не сработал
    try:
        import olefile
        if not olefile.isOleFile(str(file_path)):
            return {
                "source": str(file_path), "type": "doc", "text": None,
                "image": None, "error": "Not a valid OLE2 file", "text_quality": "none"
            }
        
        ole = olefile.OleFileIO(str(file_path))
        
        if not ole.exists('WordDocument'):
            ole.close()
            return {
                "source": str(file_path), "type": "doc", "text": None,
                "image": None, "error": "No WordDocument stream", "text_quality": "none"
            }
        
        word_stream = ole.openstream('WordDocument').read()
        
        if len(word_stream) < 32:
            ole.close()
            return {
                "source": str(file_path), "type": "doc", "text": None,
                "image": None, "error": "WordDocument too small", "text_quality": "none"
            }
        
        fib_base = struct.unpack_from('<H', word_stream, 0)[0]
        if fib_base != 0xA5EC:
            ole.close()
            return {
                "source": str(file_path), "type": "doc", "text": None,
                "image": None, "error": f"Invalid FIB signature: {hex(fib_base)}", "text_quality": "none"
            }
        
        flags = struct.unpack_from('<H', word_stream, 0x000A)[0]
        table_name = '1Table' if (flags & 0x0200) else '0Table'
        
        if not ole.exists(table_name):
            ole.close()
            return {
                "source": str(file_path), "type": "doc", "text": None,
                "image": None, "error": f"No {table_name} stream", "text_quality": "none"
            }
        
        table_stream = ole.openstream(table_name).read()
        
        clx_offset = struct.unpack_from('<I', word_stream, 0x01A2)[0]
        clx_size = struct.unpack_from('<I', word_stream, 0x01A6)[0]
        
        if clx_offset == 0 or clx_size == 0 or clx_offset + clx_size > len(table_stream):
            ole.close()
            return {
                "source": str(file_path), "type": "doc", "text": None,
                "image": None, "error": "Invalid CLX offset/size", "text_quality": "none"
            }
        
        clx = table_stream[clx_offset:clx_offset + clx_size]
        
        pos = 0
        text_parts = []
        
        while pos < len(clx):
            clx_type = clx[pos]
            
            if clx_type == 0x01:
                pos += 1
                cb = struct.unpack_from('<H', clx, pos)[0]
                pos += 2 + cb
                continue
            
            elif clx_type == 0x02:
                pos += 1
                cb = struct.unpack_from('<I', clx, pos)[0]
                pos += 4
                
                piece_table = clx[pos:pos + cb]
                
                n_pieces = (cb - 4) // 12
                if n_pieces <= 0:
                    break
                
                cp_offsets = []
                for i in range(n_pieces + 1):
                    cp = struct.unpack_from('<I', piece_table, i * 4)[0]
                    cp_offsets.append(cp)
                
                pd_start = (n_pieces + 1) * 4
                
                for i in range(n_pieces):
                    cp_start = cp_offsets[i]
                    cp_end = cp_offsets[i + 1]
                    
                    pd_offset = pd_start + i * 8
                    fc = struct.unpack_from('<I', piece_table, pd_offset + 2)[0]
                    
                    is_unicode = (fc & 0x40000000) == 0
                    fc_real = fc & 0x3FFFFFFF
                    
                    char_count = cp_end - cp_start
                    
                    if is_unicode:
                        byte_offset = fc_real
                        byte_count = char_count * 2
                        if byte_offset + byte_count <= len(word_stream):
                            text_bytes = word_stream[byte_offset:byte_offset + byte_count]
                            try:
                                text_parts.append(text_bytes.decode('utf-16-le', errors='ignore'))
                            except:
                                pass
                    else:
                        byte_offset = fc_real // 2
                        byte_count = char_count
                        if byte_offset + byte_count <= len(word_stream):
                            text_bytes = word_stream[byte_offset:byte_offset + byte_count]
                            try:
                                text_parts.append(text_bytes.decode('cp1251', errors='ignore'))
                            except:
                                pass
                
                break
            else:
                break
        
        ole.close()
        
        full_text = ''.join(text_parts)
        full_text = full_text.replace('\r', '\n').replace('\x07', ' ').replace('\x0b', '\n')
        full_text = ''.join(c for c in full_text if ord(c) >= 32 or c in '\n\t')
        full_text = '\n'.join(line.strip() for line in full_text.split('\n') if line.strip())
        
        if full_text.strip():
            base_result['text'] = full_text.strip()
            base_result['text_quality'] = assess_text_quality(full_text)
            return base_result
        
        base_result['error'] = "Empty after parsing"
        return base_result
            
    except ImportError:
        base_result['error'] = "olefile not available"
        return base_result
    except Exception as e:
        logger.warning(f"olefile extraction failed for {file_path.name}: {e}")
        base_result['error'] = f"OLE2 extraction error: {e}"
        return base_result
