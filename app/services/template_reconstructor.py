




# app/services/template_reconstructor.py - ENHANCED FOR DYNAMIC STRUCTURE GENERATION
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class TemplateReconstructor:
    """Reconstruct complete mapping templates with dynamic structure generation"""
    
    def reconstruct_template_from_chunks(self, search_results: List[Dict]) -> Optional[Dict[str, Any]]:
        """Reconstruct complete template from multiple related chunks"""
        
        if not search_results:
            return None
        
        # Group chunks by template name
        chunks_by_template = {}
        for result in search_results:
            try:
                # Handle Milvus Result objects properly
                if hasattr(result, 'entity'):
                    entity = result.entity
                    similarity = 1 - float(result.score) if hasattr(result, 'score') else 0.5
                else:
                    entity = result
                    similarity = 0.5
                
                # Extract template name
                template_name = self._safe_get_attribute(entity, 'template_name', '')
                
                if not template_name:
                    logger.warning("No template name found in search result, skipping")
                    continue
                
                if template_name not in chunks_by_template:
                    chunks_by_template[template_name] = []
                
                # Extract chunk data
                chunk_data = {
                    'chunk_type': self._safe_get_attribute(entity, 'chunk_type', ''),
                    'similarity': similarity,
                    'metadata': self._safe_get_attribute(entity, 'metadata', {}),
                    'text': self._safe_get_attribute(entity, 'text', ''),
                    'source_format': self._safe_get_attribute(entity, 'source_format', ''),
                    'target_format': self._safe_get_attribute(entity, 'target_format', ''),
                    'mapping_type': self._safe_get_attribute(entity, 'mapping_type', ''),
                    'carrier_code': self._safe_get_attribute(entity, 'carrier_code', ''),
                    'route_pattern': self._safe_get_attribute(entity, 'route_pattern', ''),
                    'complexity_score': self._safe_get_attribute(entity, 'complexity_score', 0.5)
                }
                
                chunks_by_template[template_name].append(chunk_data)
                
            except Exception as e:
                logger.warning(f"Error processing search result: {e}")
                continue
        
        if not chunks_by_template:
            logger.warning("No valid chunks found for template reconstruction")
            return None
        
        # Find template with highest average similarity
        best_template_name = None
        best_avg_similarity = 0
        
        for template_name, chunks in chunks_by_template.items():
            if chunks:
                avg_similarity = sum(chunk['similarity'] for chunk in chunks) / len(chunks)
                logger.info(f"Template candidate: {template_name}, avg similarity: {avg_similarity:.3f}, chunks: {len(chunks)}")
                
                if avg_similarity > best_avg_similarity:
                    best_avg_similarity = avg_similarity
                    best_template_name = template_name
        
        if not best_template_name or best_avg_similarity < 0.4:
            logger.warning(f"No high-quality template match found. Best similarity: {best_avg_similarity:.3f}")
            return None
        
        logger.info(f"Reconstructing template: {best_template_name} (similarity: {best_avg_similarity:.3f})")
        
        # Reconstruct template from best matching chunks
        return self._reconstruct_from_chunk_metadata(
            chunks_by_template[best_template_name], best_template_name, best_avg_similarity
        )
    
    def _safe_get_attribute(self, entity, attr_name: str, default=None):
        """Safely get attribute from entity using multiple access methods"""
        try:
            # Try direct attribute access first
            if hasattr(entity, attr_name):
                return getattr(entity, attr_name)
            
            # Try dictionary-style access
            if hasattr(entity, 'get') and callable(entity.get):
                return entity.get(attr_name, default)
            elif isinstance(entity, dict):
                return entity.get(attr_name, default)
            
            # Try as dict keys if entity is dict-like
            if hasattr(entity, '__getitem__'):
                try:
                    return entity[attr_name]
                except (KeyError, TypeError):
                    pass
            
            return default
        except Exception as e:
            logger.debug(f"Error accessing attribute {attr_name}: {e}")
            return default
    
    def _reconstruct_from_chunk_metadata(self, chunks: List[Dict], template_name: str, confidence: float) -> Dict[str, Any]:
        """Reconstruct template with complete business logic and dynamic structure"""
        
        complete_context_chunk = None
        transformation_chunk = None
        
        # Find the best chunks
        for chunk in chunks:
            chunk_type = chunk['chunk_type']
            if chunk_type == 'complete_mapping_context':
                complete_context_chunk = chunk
            elif chunk_type == 'transformation_logic':
                transformation_chunk = chunk
        
        if not complete_context_chunk:
            logger.warning("No complete context chunk found, using enhanced fallback")
            return self._create_enhanced_dynamic_template(template_name, confidence)
        
        try:
            # Extract metadata
            metadata = complete_context_chunk['metadata']
            
            # Create comprehensive template structure
            template = {
                'name': template_name,
                'localContext': self._reconstruct_comprehensive_local_context(),
                'targetTreeNode': self._create_dynamic_target_tree_structure(),
                'reconstruction_confidence': confidence,
                'source_chunks': len(chunks),
                'template_metadata': metadata,
                'chunk_types_used': [chunk['chunk_type'] for chunk in chunks],
                'dynamic_structure': True,
                'business_logic_included': True
            }
            
            logger.info(f"Successfully reconstructed enhanced template: {template_name} with {len(chunks)} chunks")
            return template
            
        except Exception as e:
            logger.error(f"Error reconstructing template {template_name}: {e}")
            return self._create_enhanced_dynamic_template(template_name, confidence)
    
    def _create_dynamic_target_tree_structure(self) -> Dict[str, Any]:
        """Create dynamic target tree structure that can be adapted to any input"""
        
        return {
            'name': 'root',
            'children': [
                # Core fields with dynamic mapping capability
                {'name': 'masterAirWayBillNumber', 'dynamic_mapping': True},
                {'name': 'trackingNumber', 'dynamic_mapping': True, 'transformation': 'remove_hyphens'},
                {'name': 'consigneeName', 'plain': True},
                {'name': 'shipperName', 'plain': True},
                {'name': 'retrievedAt', 'plain': True},
                
                # Dynamic array structure for house AWBs
                {
                    'name': 'houseAirWayBillNumbers',
                    'type': 'ar',
                    'children': [{
                        'name': '[0]',
                        'type': 'ac',
                        'children': [
                            {'name': 'origin', 'dynamic_mapping': True},
                            {'name': 'destination', 'dynamic_mapping': True},
                            {'name': 'number', 'dynamic_mapping': True},
                            {'name': 'quantity', 'dynamic_mapping': True, 'default': '0'},
                            {
                                'name': 'weight',
                                'children': [
                                    {'name': 'value', 'dynamic_mapping': True},
                                    {'name': 'unit', 'dynamic_mapping': True}
                                ]
                            }
                        ]
                    }]
                },
                
                # Core location fields
                {'name': 'origin', 'dynamic_mapping': True},
                {'name': 'destination', 'dynamic_mapping': True},
                {'name': 'routes', 'type': 'ar', 'dynamic_mapping': True},
                
                # Weight and pieces with dynamic mapping
                {'name': 'totalPieces', 'dynamic_mapping': True, 'transformation': 'to_number'},
                {
                    'name': 'totalWeight',
                    'children': [
                        {'name': 'value', 'dynamic_mapping': True, 'transformation': 'to_number'},
                        {'name': 'unit', 'dynamic_mapping': True}
                    ]
                },
                {
                    'name': 'totalVolume',
                    'children': [
                        {'name': 'value', 'dynamic_mapping': True, 'default': '0'},
                        {'name': 'unit', 'dynamic_mapping': True}
                    ]
                },
                
                # Business logic calculation block
                {
                    'name': 'CALC:Events&Stops',
                    'type': 'c',
                    'value': self._get_comprehensive_business_logic(),
                    'references': [
                        {'jsonId': 7, 'path': 'root.shipments[0].events', 'var': 'eventLoop', 'text': False},
                        {'jsonId': 8, 'path': 'root.shipments[0].routes', 'var': 'stopLoop', 'text': False}
                    ],
                    'dynamic_array_processing': True
                },
                
                # Additional information with dynamic special handling
                {
                    'name': 'additionalInformation',
                    'children': [{
                        'name': 'specialHandling',
                        'type': 'ar',
                        'children': [{
                            'name': '[0]',
                            'value': 'it',
                            'type': 'ac',
                            'looper': {'loopStatement': 'specialHandlingList.each'}
                        }]
                    }]
                },
                
                # Flight plan with comprehensive route processing
                {
                    'name': 'flightPlan',
                    'children': [{
                        'name': 'segmentDetails',
                        'type': 'ar',
                        'children': [{
                            'name': '[0]',
                            'type': 'ac',
                            'looper': {'loopStatement': 'stopList.each'},
                            'codeValue': self._get_comprehensive_route_processing(),
                            'children': self._get_dynamic_route_fields()
                        }]
                    }]
                },
                
                # Milestones with comprehensive event processing
                {
                    'name': 'milestones',
                    'type': 'ar',
                    'children': [{
                        'name': '[0]',
                        'type': 'ac',
                        'looper': {'loopStatement': 'eventList.each'},
                        'codeValue': self._get_comprehensive_event_processing(),
                        'children': self._get_dynamic_event_fields()
                    }]
                }
            ]
        }
    
    def _get_dynamic_route_fields(self) -> List[Dict[str, Any]]:
        """Get comprehensive dynamic route field mappings"""
        return [
            {'name': 'origin', 'value': 'origin'},
            {'name': 'destination', 'value': 'destination'},
            {'name': 'totalPieces', 'value': 'totalPieces'},
            {'name': 'totalWeight', 'children': [
                {'name': 'value', 'value': 'totalWeightValue'},
                {'name': 'unit', 'value': 'totalWeightUnit'}
            ]},
            {'name': 'totalVolume', 'children': [
                {'name': 'value', 'value': 'totalVolumeValue'},
                {'name': 'unit', 'value': 'totalVolumeUnit'}
            ]},
            {'name': 'vehicleNumber', 'value': 'vehicleNumber'},
            {'name': 'transportMode', 'value': 'transportMode'},
            {'name': 'scheduledDepartureTime', 'value': 'scheduledDepartureTime'},
            {'name': 'actualDepartureTime', 'value': 'actualDepartureTime'},
            {'name': 'scheduledArrivalTime', 'value': 'scheduledArrivalTime'},
            {'name': 'actualArrivalTime', 'value': 'actualArrivalTime'},
            {'name': 'estimatedDepartureTime', 'value': 'estimatedDepartureTime'},
            {'name': 'estimatedArrivalTime', 'value': 'estimatedArrivalTime'},
            {'name': 'timeZone', 'value': 'timeZone'},
            {'name': 'bookingStatus', 'value': 'bookingStatus'},
            {'name': 'segmentIndex', 'value': 'segmentIndex'},
            {'name': 'bookingStatusDescription', 'value': 'bookingStatusDescription'}
        ]
    
    def _get_dynamic_event_fields(self) -> List[Dict[str, Any]]:
        """Get comprehensive dynamic event field mappings"""
        return [
            {'name': 'eventType', 'value': 'eventType'},
            {'name': 'carrierEventCode', 'value': 'carrierEventCode'},
            {'name': 'eventCode', 'value': 'eventCode'},
            {'name': 'description', 'value': 'description'},
            {'name': 'station', 'value': 'station'},
            {'name': 'totalPieces', 'value': 'totalPieces'},
            {'name': 'totalWeight', 'children': [
                {'name': 'value', 'value': 'totalWeightValue'},
                {'name': 'unit', 'value': 'totalWeightUnit'}
            ]},
            {'name': 'totalVolume', 'children': [
                {'name': 'value', 'value': 'totalVolumeValue'},
                {'name': 'unit', 'value': 'totalVolumeUnit'}
            ]},
            {'name': 'timeZone', 'value': 'timeZone'},
            {'name': 'eventTime', 'value': 'eventTime'},
            {'name': 'sequence', 'value': 'sequence'},
            {'name': 'estimatedDepartureTime', 'value': 'estimatedDepartureTime'},
            {'name': 'estimatedArrivalTime', 'value': 'estimatedArrivalTime'},
            {'name': 'actualDepartureTime', 'value': 'actualDepartureTime'},
            {'name': 'scheduledArrivalTime', 'value': 'scheduledArrivalTime'},
            {'name': 'scheduledDepartureTime', 'value': 'scheduledDepartureTime'},
            {'name': 'actualArrivalTime', 'value': 'actualArrivalTime'},
            {
                'name': 'ulds',
                'type': 'ar',
                'children': [{
                    'name': '[0]',
                    'type': 'ac',
                    'looper': {'loopStatement': 'it.uldList.each'},
                    'codeValue': self._get_uld_processing(),
                    'children': [
                        {'name': 'type', 'value': 'type'},
                        {'name': 'number', 'value': 'number'},
                        {'name': 'serialNumber', 'value': 'serialNumber'},
                        {'name': 'ownerCode', 'value': 'ownerCode'},
                        {'name': 'quantity', 'value': 'quantity'},
                        {'name': 'weightOfULDContents', 'children': [
                            {'name': 'value', 'value': 'weightOfULDContentsValue'},
                            {'name': 'unit', 'value': 'weightOfULDContentsUnit'}
                        ]}
                    ]
                }]
            },
            {'name': 'vehicleNumber', 'value': 'vehicleNumber'},
            {'name': 'transportMode', 'value': 'transportMode'},
            {'name': 'departureLocation', 'value': 'departureLocation'},
            {'name': 'arrivalLocation', 'value': 'arrivalLocation'},
            {'name': 'transferManifestNumber', 'value': 'transferManifestNumber'},
            {'name': 'transferredFromName', 'value': 'transferredFromName'},
            {'name': 'receivedFromName', 'value': 'receivedFromName'},
            {'name': 'deliveryToName', 'value': 'deliveryToName'},
            {'name': 'notificationToName', 'value': 'notificationToName'},
            {'name': 'receivingCarrier', 'value': 'receivingCarrier'},
            {'name': 'transferringCarrier', 'value': 'transferringCarrier'},
            {'name': 'discrepencyCode', 'value': 'discrepencyCode'}
        ]
    
    def _reconstruct_comprehensive_local_context(self) -> Dict[str, Any]:
        """Reconstruct comprehensive local context with all business classes"""
        
        return {
            'globalVariables': [],
            'functions': [],
            'lookupTables': [],
            'classes': [
                {
                    'name': 'Stop',
                    'value': '''class Stop {
    String origin = ""
    String destination = ""
    String totalPieces = ""
    String totalWeightValue = ""
    String totalWeightUnit = ""
    String totalVolumeValue = ""
    String totalVolumeUnit = ""
    String vehicleNumber = ""
    String transportMode = ""
    String scheduledDepartureTime = ""
    String actualDepartureTime = ""
    String scheduledArrivalTime = ""
    String actualArrivalTime = ""
    String estimatedDepartureTime = ""
    String estimatedArrivalTime = ""
    String timeZone = ""
    String bookingStatus = ""
    String segmentIndex = ""
    String bookingStatusDescription = ""
}''',
                    'shortValue': 'class Stop {...}'
                },
                {
                    'name': 'Event',
                    'value': '''class Event {
    String eventType = ""
    String carrierEventCode = ""
    String eventCode = ""
    String description = ""
    String station = ""
    String totalPieces = ""
    String totalWeightValue = ""
    String totalWeightUnit = ""
    String totalVolumeValue = ""
    String totalVolumeUnit = ""
    String timeZone = ""
    String eventTime = ""
    String sequence = ""
    String estimatedDepartureTime = ""
    String estimatedArrivalTime = ""
    String actualDepartureTime = ""
    String scheduledArrivalTime = ""
    String scheduledDepartureTime = ""
    String actualArrivalTime = ""
    String vehicleNumber = ""
    String transportMode = ""
    String departureLocation = ""
    String arrivalLocation = ""
    String transferManifestNumber = ""
    String transferredFromName = ""
    String receivedFromName = ""
    String deliveryToName = ""
    String notificationToName = ""
    String receivingCarrier = ""
    String transferringCarrier = ""
    String discrepencyCode = ""
    LinkedList<Uld> uldList = new LinkedList<Uld>()
}''',
                    'shortValue': 'class Event {...}'
                },
                {
                    'name': 'Uld',
                    'value': '''class Uld {
    String type = ""
    String number = ""
    String serialNumber = ""
    String ownerCode = ""
    String quantity = ""
    String weightOfULDContentsValue = ""
    String weightOfULDContentsUnit = ""
}''',
                    'shortValue': 'class Uld {...}'
                },
                {
                    'name': 'MapperUtility',
                    'value': '''class MapperUtility {
    public static boolean isNullOrEmpty(def item){
        if(item == "" || item == null || item == "null")
        {
        return true
        }
        return false
    }
    public static def convertToDouble(def item)
    {   
        def output
            if(isNullOrEmpty(item))
            {
            output= null
            }
            else
            {
            output = item.toDouble()
            }
        return output
    }
}''',
                    'shortValue': 'class MapperUtility {...}'
                }
            ]
        }
    
    def _get_comprehensive_business_logic(self) -> str:
        """Get comprehensive business logic for processing events and stops"""
        
        return '''LinkedList<Stop> stopList = new LinkedList<Stop>()
LinkedList<Event> eventList = new LinkedList<Event>()
LinkedList<String> specialHandlingList = new LinkedList<String>()

def CarrierEventCodeMap = [:]
def CarrierDescriptionCodeMap = [:]

// Enhanced carrier event code mappings
CarrierEventCodeMap["NFD"] = "NFD"
CarrierEventCodeMap["RCF"] = "RCF"
CarrierEventCodeMap["DEP"] = "P1"
CarrierEventCodeMap["RCS"] = "RCS"
CarrierEventCodeMap["FOH"] = "FOH"
CarrierEventCodeMap["DLV"] = "DLV"
CarrierEventCodeMap["AWD"] = "AWD"
CarrierEventCodeMap["TFD"] = "AN"
CarrierEventCodeMap["AWR"] = "AWR"
CarrierEventCodeMap["CCD"] = "CCD"
CarrierEventCodeMap["CRC"] = "CRC"
CarrierEventCodeMap["DIS"] = "DIS"
CarrierEventCodeMap["BKD"] = "DR"
CarrierEventCodeMap["BKG"] = "DR"
CarrierEventCodeMap["FWB"] = "R1"
CarrierEventCodeMap["FFM"] = "MAN"
CarrierEventCodeMap["MAN"] = "MAN"
CarrierEventCodeMap["PRE"] = "PRE"
CarrierEventCodeMap["RCT"] = "R1"
CarrierEventCodeMap["TRM"] = "TRM"
CarrierEventCodeMap["ARR"] = "X4"
CarrierEventCodeMap["KK"] = "DR"

// Enhanced description mappings
CarrierDescriptionCodeMap["delivered"] = "DLV"
CarrierDescriptionCodeMap["document delivered to forwarder"] = "AWD"
CarrierDescriptionCodeMap["freight accepted at airport"] = "RCF"
CarrierDescriptionCodeMap["arrived"] = "X4"
CarrierDescriptionCodeMap["departed"] = "P1"
CarrierDescriptionCodeMap["departed on flight"] = "P1"
CarrierDescriptionCodeMap["manifested on flight"] = "MAN"
CarrierDescriptionCodeMap["airline received"] = "RCS"
CarrierDescriptionCodeMap["received from shipper"] = "RCS"
CarrierDescriptionCodeMap["freight on hand"] = "FOH"
CarrierDescriptionCodeMap["discrepancy"] = "DIS"
CarrierDescriptionCodeMap["freight received from airline"] = "R1"
CarrierDescriptionCodeMap["freight ready for pick up"] = "NFD"
CarrierDescriptionCodeMap["freight transferred to airline"] = "AN"
CarrierDescriptionCodeMap["documents received"] = "AWR"
CarrierDescriptionCodeMap["other information"] = "OCI"
CarrierDescriptionCodeMap["custom cleared"] = "CCD"
CarrierDescriptionCodeMap["freight to be transferred to airline"] = "TRM"
CarrierDescriptionCodeMap["prepared for loading"] = "PRE"
CarrierDescriptionCodeMap["booked"] = "DR"

// Process routes into stops
if(stopLoop != null && stopLoop.size() > 0) {
    stopLoop.eachWithIndex{ st, st_index->
        Stop stopObj = new Stop()
        stopObj.origin = st.origin?.toString() ?: ""
        stopObj.destination = st.destination?.toString() ?: ""
        stopObj.totalPieces = st.totalPieces?.toString() ?: ""
        
        // Handle different weight structures dynamically
        if(st.weight?.value) {
            stopObj.totalWeightValue = st.weight.value.toString()
            stopObj.totalWeightUnit = st.weight.unit?.toString() ?: ""
        } else if(st.totalWeight?.value) {
            stopObj.totalWeightValue = st.totalWeight.value.toString()
            stopObj.totalWeightUnit = st.totalWeight.unit?.toString() ?: ""
        }
        
        stopObj.vehicleNumber = st.flightNum?.toString() ?: ""
        stopObj.segmentIndex = st.routeIndex?.toString() ?: "${st_index+1}"
        stopObj.transportMode = st.transportMode?.toString() ?: ""
        stopObj.scheduledDepartureTime = st.departureDateExpected?.toString() ?: ""
        stopObj.actualDepartureTime = st.departureDate?.toString() ?: ""
        stopObj.scheduledArrivalTime = st.arrivalDateExpected?.toString() ?: ""
        stopObj.actualArrivalTime = st.arrivalDate?.toString() ?: ""
        stopObj.estimatedDepartureTime = st.estimatedDepartureDate?.toString() ?: ""
        stopObj.estimatedArrivalTime = st.estimatedArrivalDate?.toString() ?: ""
        stopObj.timeZone = ""
        stopObj.bookingStatus = ""
        stopObj.bookingStatusDescription = st.additionalInfo?.description?.toString() ?: ""
        
        stopList.add(stopObj)
    }
}

// Process events
if(eventLoop != null && eventLoop.size() > 0) {
    eventLoop.eachWithIndex{ et, et_index->
        Event eventObj = new Event()
        
        eventObj.eventType = et.eventQualifier?.toString() ?: ""
        eventObj.description = et.status?.toString() ?: ""
        eventObj.carrierEventCode = et.eventCode?.toString() ?: ""
        eventObj.station = et.location?.name?.toString() ?: ""
        
        // Map event codes
        eventObj.eventCode = CarrierEventCodeMap.get(eventObj.carrierEventCode)
        if(MapperUtility.isNullOrEmpty(eventObj.eventCode)) {
            eventObj.eventCode = CarrierDescriptionCodeMap.get(eventObj.description?.toLowerCase())
        }
        
        // Special handling for RCF -> RCS conversion
        if(eventObj.eventCode == "RCF" && eventObj.station?.toUpperCase() == var1?.toUpperCase()) {
            eventObj.eventCode = "RCS"
        }
        
        // Handle additional info dynamically
        if(et.additionalInfo) {
            eventObj.totalPieces = et.additionalInfo.totalPieces?.toString() ?: ""
            
            if(et.additionalInfo.totalWeight) {
                eventObj.totalWeightValue = et.additionalInfo.totalWeight.value?.toString() ?: ""
                eventObj.totalWeightUnit = et.additionalInfo.totalWeight.unit?.toString() ?: ""
            }
            
            eventObj.vehicleNumber = et.additionalInfo.vehicleNumber?.toString() ?: ""
            eventObj.departureLocation = et.additionalInfo.origin?.toString() ?: ""
            eventObj.arrivalLocation = et.additionalInfo.destination?.toString() ?: ""
            eventObj.estimatedDepartureTime = et.additionalInfo.estimatedDepartureTime?.toString() ?: ""
            eventObj.estimatedArrivalTime = et.additionalInfo.estimatedArrivalTime?.toString() ?: ""
            eventObj.actualDepartureTime = et.additionalInfo.departureDate?.toString() ?: ""
            eventObj.scheduledArrivalTime = et.additionalInfo.arrivalDateExpected?.toString() ?: ""
            eventObj.scheduledDepartureTime = et.additionalInfo.departureDateExpected?.toString() ?: ""
            eventObj.actualArrivalTime = et.additionalInfo.arrivalDate?.toString() ?: ""
        }
        
        eventObj.eventTime = et.eventTime?.toString() ?: ""
        eventObj.sequence = "${et_index+1}"
        
        // Handle ULD processing
        if(et.additionalInfo?.ulds) {
            et.additionalInfo.ulds.each{ uld_it ->
                Uld uldObj = new Uld()
                uldObj.type = uld_it.type?.toString() ?: ""
                uldObj.number = uld_it.number?.toString() ?: ""
                uldObj.serialNumber = uld_it.serialNumber?.toString() ?: ""
                uldObj.ownerCode = uld_it.ownerCode?.toString() ?: ""
                uldObj.quantity = uld_it.quantity?.toString() ?: ""
                if(uld_it.weightOfULDContents) {
                    uldObj.weightOfULDContentsValue = uld_it.weightOfULDContents.value?.toString() ?: ""
                    uldObj.weightOfULDContentsUnit = uld_it.weightOfULDContents.unit?.toString() ?: ""
                }
                eventObj.uldList.add(uldObj)
            }
        }
        
        // Special processing for manifested events
        if(eventObj.description?.toLowerCase()?.contains("manifested")) {
            eventObj.departureLocation = eventObj.station
            if(eventObj.description?.toLowerCase()?.contains("to") && eventObj.description?.toLowerCase()?.contains(" ")) {
                def arr = eventObj.description.split(" ")
                eventObj.arrivalLocation = arr[arr.size()-1].trim()
            }
        }
        
        eventList.add(eventObj)
    }
}

// Set sequence numbers
def seq = 1
eventList.each {
    it.sequence = seq
    seq++
}'''
    
    def _get_comprehensive_route_processing(self) -> str:
        """Get comprehensive route processing code"""
        return '''String origin = it.origin
String destination = it.destination
def totalPieces = MapperUtility.convertToDouble(it.totalPieces)
def totalWeightValue = MapperUtility.convertToDouble(it.totalWeightValue)
String totalWeightUnit = it.totalWeightUnit
def totalVolumeValue = MapperUtility.convertToDouble(it.totalVolumeValue)
String totalVolumeUnit = it.totalVolumeUnit
String vehicleNumber = it.vehicleNumber
String transportMode = it.transportMode
String scheduledDepartureTime = it.scheduledDepartureTime
String actualDepartureTime = it.actualDepartureTime
String scheduledArrivalTime = it.scheduledArrivalTime
String actualArrivalTime = it.actualArrivalTime
String estimatedDepartureTime = it.estimatedDepartureTime
String estimatedArrivalTime = it.estimatedArrivalTime
String timeZone = it.timeZone
String bookingStatus = it.bookingStatus
def segmentIndex = MapperUtility.convertToDouble(it.segmentIndex)
String bookingStatusDescription = it.bookingStatusDescription'''
    
    def _get_comprehensive_event_processing(self) -> str:
        """Get comprehensive event processing code"""
        return '''String eventType = it.eventType
String carrierEventCode = it.carrierEventCode
String eventCode = it.eventCode
String description = it.description
String station = it.station
def totalPieces = MapperUtility.convertToDouble(it.totalPieces)
def totalWeightValue = MapperUtility.convertToDouble(it.totalWeightValue)
String totalWeightUnit = it.totalWeightUnit
def totalVolumeValue = MapperUtility.convertToDouble(it.totalVolumeValue)
String totalVolumeUnit = it.totalVolumeUnit
String timeZone = it.timeZone
String eventTime = it.eventTime
def sequence = MapperUtility.convertToDouble(it.sequence)
String estimatedDepartureTime = it.estimatedDepartureTime
String estimatedArrivalTime = it.estimatedArrivalTime
String actualDepartureTime = it.actualDepartureTime
String scheduledArrivalTime = it.scheduledArrivalTime
String scheduledDepartureTime = it.scheduledDepartureTime
String actualArrivalTime = it.actualArrivalTime
String vehicleNumber = it.vehicleNumber
String transportMode = it.transportMode
String departureLocation = it.departureLocation
String arrivalLocation = it.arrivalLocation
String transferManifestNumber = it.transferManifestNumber
String transferredFromName = it.transferredFromName
String receivedFromName = it.receivedFromName
String deliveryToName = it.deliveryToName
String notificationToName = it.notificationToName
String receivingCarrier = it.receivingCarrier
String transferringCarrier = it.transferringCarrier
String discrepencyCode = it.discrepencyCode'''
    
    def _get_uld_processing(self) -> str:
        """Get ULD processing code"""
        return '''String number = it.number
String type = it.type
String serialNumber = it.serialNumber
String ownerCode = it.ownerCode
String quantity = it.quantity
def weightOfULDContentsValue = MapperUtility.convertToDouble(it.weightOfULDContentsValue)
String weightOfULDContentsUnit = it.weightOfULDContentsUnit'''
    
    def _create_enhanced_dynamic_template(self, template_name: str, confidence: float) -> Dict[str, Any]:
        """Create enhanced dynamic template for fallback"""
        
        return {
            'name': f"Enhanced_Dynamic_{template_name}",
            'localContext': self._reconstruct_comprehensive_local_context(),
            'targetTreeNode': self._create_dynamic_target_tree_structure(),
            'reconstruction_confidence': confidence,
            'source_chunks': 0,
            'fallback_template': True,
            'enhanced_fallback': True,
            'dynamic_structure': True,
            'business_logic_included': True
        }